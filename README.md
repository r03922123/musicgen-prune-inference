# musicgen-prune-mlx

Best-of-N MusicGen with **checkpoint pruning**, reimplemented in [MLX](https://github.com/ml-explore/mlx) on an M1 MacBook Air for $0.

The idea: instead of generating 8 full candidates and keeping the best, score every
candidate's clip-so-far with CLAP at fixed checkpoints and drop the weakest half
(8 → 4 → 2 → 1). The question this repo answers: **how much wall-clock time does that
actually save, and does it cost quality?**

## Why this exists

I built this pruning scheme as a prototype at KKCompany and measured a large
wall-clock speedup. The naive compute estimate, however, predicts much less:

```
baseline  8 candidates × 3 segments          = 24 candidate-segments
pruned    8 + 4 + 2                          = 14 candidate-segments   → 1.71x less compute
```

And wall-clock can only follow compute if per-step decode time actually shrinks with
batch size. Autoregressive decoding is often memory-bound, where it doesn't. This repo
measures that directly instead of guessing.

## Run

```bash
uv sync
uv run python prune.py          # self-check, tiny random model, seconds, no downloads
uv run python bench.py steps    # timing decomposition, musicgen-small shapes, ~minutes, no downloads
uv run python bench.py run      # real musicgen-small + CLAP, 20 prompts (~3.5 GB of weights)
```

## What each file does

| File | What it shows |
|---|---|
| `prune.py` | Batched generation that removes pruned rows from the token buffer **and the KV cache**. Sampling noise is keyed by each candidate's original row id, so pruning never changes a survivor's tokens. The self-check proves that (token-for-token match with the unpruned run) and has a mutation-tested failure mode. |
| `bench.py steps` | Random weights with musicgen-small's real shapes (timing depends on shapes, not values). Measures ms/step at batch 8/4/2/1 per third of the clip, predicts the pruned wall-clock from it, then measures the real pruned loop. |
| `bench.py run` | Real weights. Per prompt: timed baseline vs timed pruned run, plus the proxy-quality metrics: Spearman(checkpoint score, final score), false-kill rate of the top-2 finals, whether pruning picked the same winner, and the CLAP score lost. |
| `vendor/` | Unmodified MusicGen / EnCodec / T5 from [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/musicgen) (MIT, see `vendor/LICENSE-mlx-examples`). |

## Results

### `bench.py steps`: musicgen-small shapes, float16, 1500 steps (30 s), M1 MacBook Air 16 GB

ms per decode step, by third of the clip:

| batch (rows with CFG) | 0–10 s | 10–20 s | 20–30 s |
|---|---|---|---|
| 8 (16) | 52.4 | 68.3 | 86.7 |
| 4 (8)  | 38.6 | 48.9 | 56.9 |
| 2 (4)  | 31.3 | 35.1 | 38.5 |
| 1 (2)  | 27.9 | 29.8 | 31.6 |

| | speedup |
|---|---|
| naive compute estimate (24 / 14 candidate-segments) | 1.71x |
| predicted wall-clock (measured step times, 8 → 4 → 2) | 1.48x |
| **measured wall-clock, real pruned loop** (104 s → 65 s) | **1.59x** |

Two things the table shows:

1. **Halving the batch does not halve step time.** Going from 8 to 1 candidate cuts
   step time only ~1.9x (52 → 28 ms) in the first third. There is a large fixed
   per-step cost (weights streamed from memory, kernel launches) that pruning can't touch.
2. **Late steps cost more** (52 → 87 ms at batch 8) because attention reads a longer KV
   cache. That is why the measured speedup lands *above* the prediction from equal
   segments: pruning removes candidates exactly where steps are most expensive.

### `torch/bench.py steps`: real musicgen-small weights, float16, 1500 steps (30 s), Tesla T4

PyTorch + HF transformers port in `torch/` (same row-id-keyed noise and KV-cache
pruning; its self-check passes on CUDA). Run on Kaggle via `kaggle_job/`
(torch 2.11, transformers 5.17, eager mode). ms per decode step, by third of the clip:

| batch (rows with CFG) | 0–10 s | 10–20 s | 20–30 s |
|---|---|---|---|
| 8 (16) | 17.8 | 23.4 | 36.1 |
| 4 (8)  | 17.9 | 17.8 | 21.8 |
| 2 (4)  | 18.3 | 18.0 | 18.1 |
| 1 (2)  | 18.1 | 18.1 | 17.8 |

| | speedup |
|---|---|
| naive compute estimate (24 / 14 candidate-segments) | 1.71x |
| predicted wall-clock (measured step times, 8 → 4 → 2) | 1.44x |
| **measured wall-clock, real pruned loop** (38.6 s → 26.8 s) | **1.44x** |

Step time stops falling at about 18 ms per step, far above the ~2 ms needed to read the fp16 decoder
weights once on a T4, so it is most likely per-step overhead (Python + kernel launches in
the step-by-step loop; not profiled). Pruning only saves time where batch 8 rises above
that level, late in the clip. The pruned survivor also matched the batch-8 run
token-for-token at full scale in fp16.

### Upper bound: why "~5x" can't come from this schedule alone

For each segment, the pruned run's cost is at least `(b/8)` of the baseline's cost for
that segment, as long as step cost doesn't fall *faster than linearly* with batch size
(true unless the baseline batch was thrashing memory). With `b = 8, 4, 2`, the worst
segment is `2/8`, so:

```
speedup = Σ baseline_i / Σ pruned_i  ≤  8 / 2  =  4x      (for any cost growth over position)
```

With the position growth measured above and perfectly linear batch scaling, the
ceiling is **1.92x**. So a measured ~5x implies the baseline differed from the pruned
run in something other than the pruning: memory oversubscription at batch 8, a
different code path (e.g. sliding-window `extend` in one run only), or extra work.

### `bench.py run`

Not run yet (needs ~3.5 GB of weights). Fills in the quality side: does the 10 s CLAP
score predict the final one, and how often does pruning discard the eventual winner?

## Design notes

- **30 s, one context window.** MusicGen's position budget is 1500 frames (30 s at 50 Hz),
  shared between any audio prompt and new tokens. Pruning inside a single generation
  (checkpoints at 10/20/30 s) avoids the sliding-window `extend` path entirely.
- **CLAP scoring is chunked, not cropped.** `laion/clap-htsat-unfused` sees at most 10 s,
  and its processor uses `truncation="rand_trunc"`: a longer clip is scored on a *random*
  10 s crop, so two identical runs can prune different candidates. `bench.py` embeds
  every 10 s chunk and averages them per clip, which makes the score deterministic.
- **CFG doubles the batch.** Classifier-free guidance runs a conditional and an
  unconditional row per candidate, so "batch 8" is 16 rows on the GPU.
- **MLX is lazy.** Nothing executes until `mx.eval()`. Every decode step calls it before
  the clock is read, otherwise timings measure graph construction, not compute.

## Caveats

- M1 timings are not CUDA timings. The M1 has far less compute per byte of memory
  bandwidth than a datacenter GPU, so the batch-size scaling measured here does not
  transfer 1:1 to the original prototype's hardware. The **quality** metrics (`bench.py run`) are
  hardware-independent.
- `musicgen-small` (300M), not the model size used in the prototype.
