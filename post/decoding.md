---
date: October 24, 2026
author: r03922123
description: How MusicGen turns text into audio tokens (RVQ, the delay pattern, classifier-free guidance, Gumbel-max sampling), where a decoding step's time goes on a T4, how to prune takes from a running batch without changing the survivors, and what that means for a best-of-N music service.
---

# Inside MusicGen decoding: tokens, guidance and exact pruning

> **Summary.** The companion to [Judging text-to-music](../). It builds MusicGen from the waveform up, estimates and
> then measures where a decoding step's time goes, shows how to drop takes from a running batch without changing
> the ones that survive, and turns the measurements into the design of a best-of-N music service.

<!-- toc -->

## Introduction

[Best-of-N with early pruning](../#2-a-cheaper-search-early-pruning) keeps CLAP's pick on 97% of prompts at 0.75 of
the compute. This post covers the machinery underneath: what a take is, why pruning saves less time than compute,
and why pruned takes come out token-for-token identical to unpruned ones. Numbers are for MusicGen-small on a
Kaggle T4 and an Apple M1, from committed files in the
[companion repo](https://github.com/r03922123/musicgen-prune-inference).

## From waveform to tokens

Thirty seconds of audio at 32 kHz is $32{,}000 \times 30 = 960{,}000$ samples. Predicting one at a time
([WaveNet](https://arxiv.org/abs/1609.03499); van den Oord et al. 2016) means a million sequential steps per take.
The standard fix: **compress first, then model the compressed sequence.** A neural audio codec turns the waveform
into tokens, a language model generates tokens, and the codec turns them back into sound.

### Residual vector quantization

MusicGen's codec, [EnCodec](https://arxiv.org/abs/2210.13438) (Défossez et al. 2022), is an autoencoder that
downsamples by $8 \cdot 5 \cdot 4 \cdot 4 = 640$: 32,000 samples per second become **50 frames per second**, each a
vector $z$. **Residual vector quantization (RVQ)** turns each vector into $K = 4$ tokens, like giving directions in
stages (city, street, house number), each stage describing only what the previous ones left out:

```python
import numpy as np
rng = np.random.default_rng(0)
D, K, V = 128, 4, 2048                       # latent dim, codebooks, entries per codebook
books = rng.normal(size=(K, V, D)) / np.sqrt(D)
z = rng.normal(size=D)                       # one latent frame from the encoder

codes, residual = [], z.copy()
for k in range(K):
    j = np.argmin(((residual[None] - books[k]) ** 2).sum(-1))  # nearest entry to what is left
    codes.append(j)
    residual -= books[k][j]                  # the next codebook models the remainder
    print(k, j, np.linalg.norm(residual))    # error shrinks codebook by codebook
```

Codebook 1 carries the coarse content; later ones add detail.^[Real codecs learn the codebooks jointly with the
encoder and decoder; the random ones above only show the greedy residual step.] Count the savings: 4 codebooks ×
11 bits × 50 Hz $= 2{,}200$ bits per second, against $512{,}000$ for 16-bit audio, about **230× fewer**. Thirty
seconds become $1500 \times 4 = 6{,}000$ tokens.

### The delay pattern

Predict a frame's four tokens one after another (6,000 steps), or all at once (1,500 steps, but codebook 2 cannot
see codebook 1 of its own frame)? MusicGen's **delay pattern** shifts codebook $k$ right by $k - 1$ steps, then
predicts all four in parallel.

:::figure delay fallback=figures/delay.svg
**Fig. 1. Three ways to lay out four codebooks.** Each column is a decoding step, each cell a token labelled by its
frame. Switch layouts and click a token: outlined cells are what it can condition on, dimmed cells are written in
the same step and invisible to it. *Delay* needs $1500 + 3 = 1{,}503$ steps and keeps the codebook dependency.
:::

Undoing it is one line: frame $t$ of codebook $i$ sits at position $1 + t + i$.

```python
def undelay(seq, K):
    """(B, 1+S, K) delayed buffer with BOS at 0 -> (B, T, K) aligned frames."""
    T = seq.shape[1] - K
    return torch.stack([seq[:, 1 + i : 1 + i + T, i] for i in range(K)], dim=-1)
```

**In practice:** delay bugs are silent (the audio still sounds like audio, only worse), so test that `undelay`
inverts the delay before anything else runs.

## The generator

![MusicGen pipeline](figures/pipeline.svg "**Fig. 2. MusicGen, end to end.** A frozen T5 encoder reads the text; a decoder-only Transformer cross-attends to it and predicts 4 codebook tokens per step; EnCodec's decoder turns 1,500 frames into 30 s of audio. Numbers are for MusicGen-small.")

**Count the parameters.** [MusicGen](https://arxiv.org/abs/2306.05284) (Copet et al. 2023) is a decoder-only
Transformer trained with next-token cross-entropy. The small model has 24 layers of width $d = 1024$. Per layer:
self-attention $4d^2$, cross-attention $4d^2$, feed-forward $8d^2$, so $16d^2 \approx 16.8\text{M}$, times 24
$\approx 403\text{M}$, plus embeddings: **420M decoder parameters** (derived from the config; the model card says
"300M"). Every decoding step reads all of them.

**Classifier-free guidance.** Every step runs the model twice, with the text and with an empty prompt. If a token's
logit is $1.0$ without the text and $1.5$ with it, guidance with weight $w = 3$ triples the difference:
$1.0 + 3 \times 0.5 = 2.5$.

$$
\ell = \ell_u + w\,(\ell_c - \ell_u), \qquad w = 3 .
$$

This is [Ho & Salimans (2022)](https://arxiv.org/abs/2207.12598), and **it doubles the batch**: eight takes are 16
rows on the GPU.

```python
out = model(input_ids=x, encoder_hidden_states=txt, past_key_values=cache,
            use_cache=True).logits.view(2 * B, K, V).float()
logits = out[B:] + (out[:B] - out[B:]) * guidance      # rows [0,B) conditional, [B,2B) unconditional
```

**Sampling as noise plus argmax.** MusicGen keeps the 250 most likely tokens (top-k) and samples one. Instead of
`torch.multinomial`, I use the equivalent **Gumbel-max trick**: add Gumbel noise to the logits and take the largest.
Each token then wins exactly as often as softmax says.

$$
\arg\max_v \big(\ell_v + g_v\big) \;\sim\; \mathrm{softmax}(\ell), \qquad g_v = -\log(-\log u_v),\; u_v \sim \mathcal{U}(0,1).
$$

```python
def sample(logits, noise, top_k, temp):
    l = logits / temp
    kth = l.topk(top_k, dim=-1).values[..., -1:]
    l = l.masked_fill(l < kth, -math.inf)                # restrict to the top-k set
    return (l + noise).argmax(dim=-1)                    # (B, K): one token per codebook
```

Why? It separates *the randomness* from *the model*. The noise is a tensor you can label "take 3, step 712", so a
take is fully determined by its prompt and its own noise, just as serving systems give each request its own seeded
generator. Exact pruning depends on this.

## What decoding costs

**Estimate first.** To write one token, the GPU reads the whole model: 420M parameters × 2 bytes = 0.84 GB, plus
the KV cache (96 KiB per row per frame, about **2.2 GiB** for 16 rows by 30 s). A T4 reads about 320 GB/s, so the
weights take $0.84 / 320 \approx 2.6$ ms; the arithmetic ($2 \times 420\text{M} \times 16 \approx 13$ GFLOP at
65 TFLOPS) takes 0.2 ms. A step is **memory-bound**, so batch size should barely matter: the weights are read once
per step however many rows there are.^[The roofline argument: about 16 FLOP per byte of weights, far below the
T4's ~200 FLOP/byte ridge point.] Real-time playback needs a step under **20 ms**.

**Then measure.** 1,500 steps on the T4 (fp16, eager PyTorch;
`torch/results_steps_torch_small_float16_pretrained_t4.json`):

| Batch (rows with CFG) | ms/step, 0–10 s | 10–20 s | 20–30 s |
|---|---|---|---|
| 8 (16) | 17.8 | 23.4 | 36.1 |
| 4 (8) | 17.9 | 17.8 | 21.8 |
| 1 (2) | 18.1 | 18.1 | 17.8 |

The growth at batch 8 is the KV cache, as predicted. But at batch ≤ 4 every step takes **~18 ms whatever the
batch**, seven times the estimate. The profiler says why (50 steps;
`results/profile_decode_small_float16_pretrained_t4.json`):

| Batch (rows) | Wall ms/step | GPU busy ms/step | GPU share | Kernels/step | Mean kernel |
|---|---|---|---|---|---|
| 1 (2) | 18.7 | 6.5 | 35% | 498 | 13 µs |
| 8 (16) | 22.2 | 8.6 | 39% | 499 | 17 µs |

![GPU timeline](figures/timeline.svg "**Fig. 3. Where a decoding step goes (schematic).** The CPU spends the step launching about 500 tiny kernels through Python and PyTorch; the GPU finishes each in microseconds and then waits. Measured on a T4 at batch 1: 18.7 ms per step, of which the GPU is busy 6.5 ms.")

Python launches about 500 tiny kernels per step, more slowly than the GPU runs them, so the GPU idles 12.2 of every
18.7 ms. Small-batch eager decoding is **launch-bound**. So fewer rows barely help, and **pruning saves less time
than compute.** The standard fix is a **CUDA graph**: record a step once, replay it with one launch. Graphs need
fixed shapes, which matters for pruning; I have not measured one here.

## Making pruning exact

When you drop rows from a running batch, two things must stay true for the survivors:

1. **Their noise must not change.** Most code draws noise of shape `(batch, K, V)` each step. After pruning,
   survivor C moves from position 2 to 1 and gets *someone else's* noise: a different song. So I key noise by
   *(seed, step, original take id)*, drawing it for all 8 takes and gathering the survivors' rows.
2. **Their cache rows must move with them**: self- and cross-attention keys and values, conditional and
   unconditional halves together.

```python
g = gumbel(n, K, V, seed * 1_000_003 + off, dev)        # noise for ALL n original rows, keyed by step
tok = sample(logits, g[ids], top_k, temp)                # ids = original row ids of the survivors
...
if step in checkpoints:
    scores = score_fn(undelay(seq[:, : step + 1], K))
    keep = max(1, math.ceil(B * keep_frac))
    top = torch.argsort(-scores, stable=True)[:keep].sort().values
    seq, ids = seq[top], ids[top]
    rows = torch.cat([top, top + B])                     # conditional + unconditional rows
    txt = txt[rows]
    cache.batch_select_indices(rows)                     # self- AND cross-attention K/V
```

**In practice: test the property, then test that the test can fail.** The self-check requires every pruned survivor
to match its unpruned row token for token (100%, also at full scale in fp16 on the T4), then breaks the code on
purpose (position-keyed noise, wrong cache rows) and requires each break to fail.

The payoff: **generate once, unpruned, and simulate every schedule exactly from the saved scores.** A GPU experiment
per schedule becomes a millisecond loop, which is why the pruning figure in the companion post runs in your
browser. Measured for 8 → 4 → 2 (generation only; M1: `results_steps_small_float16.json`, T4: the step-time file
above):

| Hardware | Compute saved | Predicted from step times | Measured |
|---|---|---|---|
| Apple M1, MLX | 1.71× | 1.48× | 1.59× (104 s → 65 s) |
| Tesla T4, PyTorch eager | 1.71× | 1.44× | 1.44× (38.6 s → 26.8 s) |

The T4 falls short because of the launch-bound floor. **Caveat:** these use a cheap stand-in scorer; a real judge
at every checkpoint costs extra.

## System design: a best-of-N music service

A service takes a prompt and returns one song, running best-of-N with pruning behind the API. Each decision maps to
a measurement above.

![Service architecture](figures/system.svg "**Fig. 4. A best-of-N music service.** The decode loop shares GPUs through continuous batching; a checkpoint scorer decides what to drop; a final scorer picks the keeper; an offline loop replays logged requests to evaluate new scorers and schedules exactly.")

**Budget first.** On one T4, best-of-8 took 38.6 s of GPU time, pruned 26.8 s, one take about 27 s. The real
choice is **best-of-8 with pruning vs best-of-2 without**.

- **Release slots, do not shrink batches** (launch-bound). A pruned take's slot goes to another request at once
  (continuous batching, as in [vLLM](https://arxiv.org/abs/2309.06180)): pruning pays off as **throughput**.
- **Page the KV cache** (2.2 GiB). The cache limits how many takes fit; paging frees a pruned take's memory at once.
- **One CUDA graph per batch size** (the GPU idles 65% of each step), since pruning changes the batch.
- **Count the scorer** and overlap it with decoding on its own stream.
- **Keep the scorer swappable.** The schedule needs one number per partial take, so CLAP can give way to a learned
  judge.
- **Replay offline** (pruning is exact): log tokens and scores, and test new schedules on real traffic.

## Reproducibility

`torch/tests/test_post_numbers.py` checks the profiler table against the committed results and the timeline
figure against the profile; `torch/prune.py` runs the exactness self-check on a tiny random model in seconds.

## Exercises

1. **Codec arithmetic.** 9 codebooks of 1024 entries at 86 frames per second: what bitrate, and how many tokens
   for a 3-minute song?
2. **KV cache.** Write its size for $R$ rows, $T$ frames, $L$ layers, width $d$, fp16. How many 30 s takes with
   guidance fit in 16 GB beside MusicGen-small?
3. **Exactness.** In `torch/prune.py`, set `row_keyed=False` in the pruned call and explain the token-match number.

## References

1. van den Oord et al. ["WaveNet: A generative model for raw audio."](https://arxiv.org/abs/1609.03499) arXiv 2016.
2. Défossez et al. ["High fidelity neural audio compression."](https://arxiv.org/abs/2210.13438) TMLR 2023.
3. Copet et al. ["Simple and controllable music generation."](https://arxiv.org/abs/2306.05284) NeurIPS 2023.
4. Ho & Salimans. ["Classifier-free diffusion guidance."](https://arxiv.org/abs/2207.12598) arXiv 2022.
5. Kwon et al. ["Efficient memory management for LLM serving with PagedAttention."](https://arxiv.org/abs/2309.06180) SOSP 2023.
