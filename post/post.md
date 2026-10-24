---
date: October 24, 2026
author: r03922123
description: Best-of-N reranking, from LLM verifiers and DALL·E to text-to-music, and how to make it cheaper. How MusicGen turns text into audio tokens, where decoding time goes, and how to stop weak takes early, exactly and measurably. Interactive figures recompute every result from the data.
---

# Faster best-of-N for text-to-music: pruning weak takes early

> **Summary.** Generate N takes, score each with CLAP, keep the best: a standard reranking recipe that costs N times
> one take. This post traces where the recipe comes from, builds MusicGen from the waveform up, measures where
> decoding time goes, and stops weak takes partway without changing which take CLAP picks.

<!-- toc -->

## Introduction

Type "lo-fi hip hop with warm piano" into a text-to-music model eight times and you get eight different songs. Some
are lovely; some wander off or go nearly silent. A cheap fix is old and common: generate several candidates, score
each against the prompt, keep the best. This is **best-of-N reranking**, and it multiplies the cost of every request
by N. Most of that compute goes into takes that lose. **Can we tell early which takes are worth finishing, and stop
the rest, without changing the answer?**

The study is small and reproducible: MusicGen-small, 50 prompts, 8 takes each, a Kaggle T4 GPU and a laptop. Every
number comes from a committed results file in the
[companion repo](https://github.com/r03922123/musicgen-prune-inference), or is derived, and I say which.^[Five
minutes? Skip to [Fig. 4](#early-pruning), pick "orchestral build-up", press play on take D, then move the first
checkpoint from 10 s to 5 s.]

| Part | Question | Short answer |
|---|---|---|
| [Best-of-N](#best-of-n-where-the-recipe-comes-from) | Is CLAP reranking a real practice? | Yes: DALL·E reranked with CLIP, AudioLDM with CLAP. |
| [Tokens](#from-waveform-to-tokens), [generator](#the-generator) | What is generated? | 6,000 tokens per 30 s take, drawn with random noise. |
| [Cost](#what-decoding-costs) | Where does the time go? | N times the work, and the GPU mostly waits. |
| [Pruning](#early-pruning), [evaluation](#evaluating-the-pruner) | Can we stop weak takes early? | Yes, exactly: 0.75 of the compute, same pick on 97% of prompts. |

## Best-of-N: where the recipe comes from

Best-of-N has two halves: draw $N$ samples, then pick one with a **scorer** $r(x, y)$ that rates sample $y$ for
prompt $x$, higher meaning better. Four steps of history explain why this post uses CLAP as the scorer.

| When | Work | Draw | Pick with | What it showed |
|---|---|---|---|---|
| 2021 | DALL·E (text → image) | 512 images | CLIP | quality gains flatten after 32 samples |
| 2021 | Verifiers, WebGPT (LLMs) | many answers (64 in WebGPT) | learned verifier or reward model | best-of-64 preferred over one answer 68% of the time |
| 2023 | AudioLDM (text → audio) | 3 clips | CLAP | the recipe moves to audio |
| 2023 | MusicGen (text → music) | one clip | (none) | CLAP becomes a standard *evaluation* metric |
| 2025–26 | Audiobox Aesthetics, CMI-RM, TuneJury | N clips | learned music judges | gains per doubling of N shrink |

**Origin: reranking with a contrastive model.** [DALL·E](https://arxiv.org/abs/2102.12092) (Ramesh et al. 2021)
drew 512 images per caption and kept the one [CLIP](https://arxiv.org/abs/2103.00020) (Radford et al. 2021) ranked
highest. CLIP is two encoders, one for images and one for text, trained so a picture and its caption land close
together; their cosine similarity says "does this image match these words". DALL·E's own ablation found image
quality kept improving with more samples up to about 32, then gave diminishing returns.

**The name: best-of-N in LLMs.** [Cobbe et al. (2021)](https://arxiv.org/abs/2110.14168) sampled many solutions to
a math problem and let a trained **verifier** pick one. [WebGPT](https://arxiv.org/abs/2112.09332) (Nakano et al.
2021) called the same move "rejection sampling (best-of-n)" with a **reward model** as scorer, and people preferred
its best-of-64 answers to single answers 68% of the time. Since then the scorer has also been called a reward model
or a verifier; the recipe is the same.

**Precedent in audio: CLAP reranking.** [CLAP](https://arxiv.org/abs/2211.06687) (Wu et al. 2022) is CLIP for
audio. [AudioLDM](https://arxiv.org/abs/2301.12503) (Liu et al. 2023) uses it as a reranker: its code takes
`n_candidate_gen_per_text`, generates that many clips, and keeps the one CLAP scores highest, and AudioLDM and
AudioLDM 2 computed their reported metrics on the best of 3 clips. The Diffusers AudioLDM 2 pipeline does the same
with `num_waveforms_per_prompt`. Later work recomputed those metrics *without* reranking so that papers could be
compared fairly ([Haji-Ali et al. 2024](https://arxiv.org/abs/2406.19388)).

**The MusicGen era: CLAP as a metric.** [MusicGen](https://arxiv.org/abs/2306.05284) (Copet et al. 2023) draws one
clip per prompt and evaluates with a standard set: **FAD** (do the clips, as a whole, sound like real music),
**KL** (do they carry the same audio-tag concepts as reference clips), **CLAP score** (do they match the text), and
human ratings of overall quality and text relevance. No single metric covers everything, so you report several,
and people decide in the end.

**Now: learned music judges.** Newer scorers learn from human ratings rather than captions.
[Audiobox Aesthetics](https://arxiv.org/abs/2502.05139) (Tjandra et al. 2025) predicts human 1–10 scores for
production quality and enjoyment; I use its *content enjoyment* (CE) score. [CMI-RM](https://arxiv.org/abs/2603.00610)
(Ma et al. 2026) is a music reward model trained on human choices between two takes, with a head for musical quality
and one for prompt match, and its authors use it for exactly this kind of top-$k$ filtering. Best-of-N still shows
diminishing returns: in TuneJury ([Kim et al. 2026](https://arxiv.org/abs/2606.17006)) the gain per doubling fell
from +0.18–0.29 (4 → 8) to +0.06–0.14 (16 → 32). Dice show why: the best of one die averages 3.5, of two 4.47, of
four 5.24, of eight 5.72. Each doubling buys less, while the cost doubles every time.

**This post** takes the AudioLDM recipe with MusicGen's text metric: **pick the take with the highest CLAP score,
and make that cheaper.** CLAP has a practical advantage for pruning: it can score a 10 s prefix. But nothing below
depends on CLAP. The pruning schedule needs one number per partial take, so a learned judge can replace CLAP
without changing the method.

**In practice:** a reranked system's score on its own reranker is inflated by construction (the best of 8 always
beats their average), so judge the result with metrics the reranker did not use.

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

**Count the parameters.** MusicGen is a decoder-only Transformer trained with next-token cross-entropy. The small
model has 24 layers of width $d = 1024$. Per layer: self-attention $4d^2$, cross-attention $4d^2$, feed-forward
$8d^2$, so $16d^2 \approx 16.8\text{M}$, times 24 $\approx 403\text{M}$, plus embeddings: **420M decoder
parameters** (derived from the config; the model card says "300M"). Every decoding step reads all of them.

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
generator. Exact pruning depends on this. Note also that MusicGen writes **left to right in time**, 20 ms of audio
per step, so a 10 s prefix can be scored while the rest is unwritten.

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

## Early pruning

Can we stop losing takes partway without changing which take wins? This is early pruning, a cousin of beam search;
for LLMs, [Speculative Rejection](https://arxiv.org/abs/2410.20290) (Sun et al. 2024) does the same with a reward
model on partial answers.

### The prefix scorer

CLAP sees at most **10 seconds** and randomly crops anything longer, so I score each clip as 10 s chunks and average
their embeddings. A 10 s prefix is one chunk; a full 30 s take is three:

```python
W = 480_000                                              # 10 s at 48 kHz
for b, w in enumerate(wavs):
    w = resample_poly(w, 3, 2)                           # 32 kHz -> 48 kHz
    for s in range(0, max(len(w) - W // 10, 1), W):      # 10 s chunks, drop tails < 1 s
        chunks.append(w[s : s + W]); owner.append(b)
a = audio_features(chunks); a = a / a.norm(dim=-1, keepdim=True)
clip = torch.stack([a[owner == b].mean(0) for b in range(len(wavs))])
score = (clip / clip.norm(dim=-1, keepdim=True)) @ text.T
```

### Successive halving

The schedule is **successive halving**, the idea hyperparameter search uses to stop bad training runs early
([Jamieson & Talwalkar 2016](https://arxiv.org/abs/1502.07943)). Start 8 takes. At 10 s keep the best 4; at 20 s
the best 2; finish those and pick the better. Take-seconds: $8 \times 10 + 4 \times 10 + 2 \times 10 = 140$,
against $8 \times 30 = 240$, so $0.583$ of the compute. Given the scores, it is a pure function:

```python
def simulate(early, final, cps, keep_frac, horizon=30):
    """early: {t: scores of the t-second prefixes}; final: scores at the horizon."""
    n = len(final)
    alive, spent, prev = list(range(n)), 0.0, 0
    for t in cps:
        if len(alive) == 1:
            break
        spent += len(alive) * (t - prev)                  # takes alive x seconds generated
        keep = max(1, math.ceil(len(alive) * keep_frac))
        alive = sorted(sorted(alive, key=lambda r: -early[t][r])[:keep])   # stable: ties keep low ids
        prev = t
    spent += len(alive) * (horizon - prev)
    return {"survivors": alive, "winner": max(alive, key=lambda r: final[r]),
            "compute": spent / (n * horizon)}
```

:::figure pruner prompt=rt_00
**Fig. 4. Prune it yourself.** That function, on the real CLAP scores of all 400 takes. Each line is one take's
score as it is generated. At each dashed checkpoint the lowest takes are cut (×); their faded continuation would
never be generated. ● is the pruned run's pick, ◎ unpruned best-of-8's. Press a take to hear it. **Try:** on
"orchestral build-up", keep-half cuts take D at 20 s, yet D is unpruned best-of-8's pick; keep 0.65 saves it. Then
set the first checkpoint to 5 s and watch the slow-build rate collapse.
:::

### Making pruning exact

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
per schedule becomes a millisecond loop; that is why Fig. 4 runs in your browser. Measured for 8 → 4 → 2
(generation only; M1: `results_steps_small_float16.json`, T4: the step-time file above):

| Hardware | Compute saved | Predicted from step times | Measured |
|---|---|---|---|
| Apple M1, MLX | 1.71× | 1.48× | 1.59× (104 s → 65 s) |
| Tesla T4, PyTorch eager | 1.71× | 1.44× | 1.44× (38.6 s → 26.8 s) |

The T4 falls short because of the launch-bound floor. **Caveat:** these use a cheap stand-in scorer; a real judge
at every checkpoint costs extra.

## Evaluating the pruner

The test set is 40 MusicCaps captions plus 10 hand-written **slow-build** prompts (orchestral build-ups, post-rock
crescendos) that save their best material for late: a stress test on which scoring early should fail. Every
schedule is simulated from the one unpruned corpus, with 95% bootstrap intervals over prompts. Two questions:

- **Q1, fidelity:** does pruning keep the take unpruned best-of-8 would pick (the *same-winner rate*)? This is the
  goal: the same answer for less compute.
- **Q2, what the pick is worth:** is CLAP's pick better than a random take on the *other* metrics? This checks the
  recipe itself.

**Q1** ($n = 40$ MusicCaps, $n = 10$ slow builds; `results/frontier.json`):

| Schedule (first checkpoint / keep) | Compute | Same-winner, MusicCaps | Same-winner, slow builds |
|---|---|---|---|
| 5 s / 0.5 | 0.354 | 0.62 [0.47, 0.78] | 0.30 [0.00, 0.60] |
| 10 s / 0.5 | 0.583 | 0.88 [0.78, 0.97] | 0.80 [0.50, 1.00] |
| **10 s / 0.65** | **0.750** | **0.97 [0.93, 1.00]** | **1.00** |
| 10 s / 0.75 | 0.792 | 1.00 | 1.00 |

Scoring at 5 s is too early; the slow builds show it. From 10 s on, pruning keeps CLAP's choice. My cutoff
(same-winner ≥ 0.90) was chosen after seeing results, and I say so.

**Why does 10 s work so well?** Partly by construction: the 30 s score averages three 10 s chunks, and the 10 s
checkpoint *is* the first one. For our goal, reproducing CLAP's 30 s choice, that overlap is the mechanism, not a
flaw. It does mean the 0.97 says little about predicting music the scorer has not heard yet. A check committed
before computing it asks exactly that: does the take CLAP likes best on the **last** 10 s survive the 10 s/0.65
schedule? It does 0.88 vs 0.50 of the time at random on MusicCaps (`results/future_only.json`), but on slow builds
the excess over chance is only +0.30 [+0.00, +0.50].

**Q2**, the MusicGen-style check: the score of CLAP's pick minus the average of all 8 takes, on metrics CLAP did
not use. CLAP's own gain is left out, because the best of 8 always beats their average. "Best possible" picks with
the metric itself (MusicCaps, $n = 40$; `results/frontier.json`, `selection`):

| Metric | Gain of CLAP's pick vs a random take | Best possible |
|---|---|---|
| CMI-RM quality | +0.157 [+0.010, +0.300] | 0.590 |
| CMI-RM alignment | +0.110 [−0.004, +0.219] | 0.432 |
| Audiobox CE | +0.193 [−0.040, +0.422] | 1.115 |
| MuQ-MuLan | −0.008 [−0.030, +0.015] | 0.111 |

The eight takes really differ (the right column), but CLAP's pick captures a small part of it. Only one interval
excludes zero, and not after a Bonferroni correction for four tests ([−0.029, +0.340]). So **pruning makes CLAP
reranking cheaper, not better**, and the next gain comes from a better scorer, which the same schedule can use.

## System design: a best-of-N music service

A service takes a prompt and returns one song, running best-of-N with pruning behind the API. Each decision maps to
a measurement above.

![Service architecture](figures/system.svg "**Fig. 5. A best-of-N music service.** The decode loop shares GPUs through continuous batching; a checkpoint scorer decides what to drop; a final scorer picks the keeper; an offline loop replays logged requests to evaluate new scorers and schedules exactly.")

**Budget first.** On one T4, best-of-8 took 38.6 s of GPU time, pruned 26.8 s, one take about 27 s. The real
choice is **best-of-8 with pruning vs best-of-2 without**, and the Q2 table says what extra takes buy.

- **Release slots, do not shrink batches** (launch-bound). A pruned take's slot goes to another request at once
  (continuous batching, as in [vLLM](https://arxiv.org/abs/2309.06180)): pruning pays off as **throughput**.
- **Page the KV cache** (2.2 GiB). The cache limits how many takes fit; paging frees a pruned take's memory at once.
- **One CUDA graph per batch size** (the GPU idles 65% of each step), since pruning changes the batch.
- **Count the scorer** and overlap it with decoding on its own stream.
- **Keep the scorer swappable.** The schedule needs one number per partial take, so CLAP can give way to a learned
  judge such as CMI-RM or Audiobox Aesthetics.
- **Replay offline** (pruning is exact): log tokens and scores, and test new schedules on real traffic.

## Humans and what comes next

Automatic metrics each see one side of "good", so in the end you ask people. The mainstream method is **pairwise
preference**: show two takes, ask which to keep listening to, and fit a **Bradley–Terry** model, in which a score
gap becomes a win probability: if take $a$ scores 1.2 and $b$ scores 0.4, $a$ wins
$\sigma(0.8) \approx 69\%$ of the time.

$$
P(a \text{ beats } b) = \sigma\big(r_a - r_b\big), \qquad \sigma(t) = \frac{1}{1 + e^{-t}} .
$$

[Chatbot Arena](https://arxiv.org/abs/2403.04132) (Chiang et al. 2024) ranks LLMs this way, and music reward models
such as CMI-RM are trained on the same kind of choices. Match loudness, randomise order, and size the study first:
20 trials detect only preferences of about 75% or more. The repo has such a test (`torch/listen.py`); it has not
been run. Next steps: prune with a learned judge, and train the generator on preferences
([DPO](https://arxiv.org/abs/2305.18290)) so one take is good more often.

## Summary

- Best-of-N reranking is standard: CLIP in DALL·E, verifiers and reward models in LLMs, CLAP in AudioLDM. It costs N
  times one sample, and each doubling of N buys less.
- A codec turns 960,000 samples into 6,000 tokens; guidance doubles the batch; a take is a function of its prompt
  and its own noise.
- Small-batch decoding is launch-bound (GPU busy 35%), which caps what pruning saves in wall time.
- Noise keyed by take makes pruning exact, so one unpruned corpus evaluates every schedule.
- Successive halving from 10 s keeps CLAP's pick on 97% of prompts at 0.75 of the compute.
- Judge a reranked system on metrics the reranker did not use: CLAP's pick gains little on them.

**Limitations.** One model, 30 s instrumental clips, 50 prompts, 8 takes, one seed each; slow-build intervals are
wide ($n = 10$). CMI-RM builds on MuQ, so those metrics are not fully independent. No FAD (it needs a reference set
and many clips) and no human result yet.

## Exercises

1. **Codec arithmetic.** 9 codebooks of 1024 entries at 86 frames per second: what bitrate, and how many tokens
   for a 3-minute song?
2. **KV cache.** Write its size for $R$ rows, $T$ frames, $L$ layers, width $d$, fp16. How many 30 s takes with
   guidance fit in 16 GB beside MusicGen-small?
3. **Exactness.** In `torch/prune.py`, set `row_keyed=False` in the pruned call and explain the token-match number.

## Reproducibility

`torch/tests/test_post_numbers.py` checks every table against the committed results, and `post/tests/sim.test.mjs`
checks the interactive pruning figure; a mismatch fails the build. Every analysis decision, dated, is in the
[decision log](https://github.com/r03922123/musicgen-prune-inference/blob/master/tasks/decisions.md).

## Citation

> r03922123. (Oct 2026). Faster best-of-N for text-to-music: pruning weak takes early.
> https://r03922123.github.io/musicgen-prune-inference/

```text
@article{r03922123_2026_bestofn_music,
  title   = "Faster best-of-N for text-to-music: pruning weak takes early",
  author  = "r03922123",
  journal = "r03922123.github.io",
  year    = "2026",
  month   = "Oct",
  url     = "https://r03922123.github.io/musicgen-prune-inference/"
}
```

## References

1. Ramesh et al. ["Zero-shot text-to-image generation."](https://arxiv.org/abs/2102.12092) ICML 2021.
2. Radford et al. ["Learning transferable visual models from natural language supervision."](https://arxiv.org/abs/2103.00020) ICML 2021.
3. Cobbe et al. ["Training verifiers to solve math word problems."](https://arxiv.org/abs/2110.14168) arXiv 2021.
4. Nakano et al. ["WebGPT: Browser-assisted question-answering with human feedback."](https://arxiv.org/abs/2112.09332) arXiv 2021.
5. Wu et al. ["Large-scale contrastive language-audio pretraining."](https://arxiv.org/abs/2211.06687) ICASSP 2023.
6. Liu et al. ["AudioLDM: Text-to-audio generation with latent diffusion models."](https://arxiv.org/abs/2301.12503) ICML 2023.
7. Haji-Ali et al. ["Taming data and transformers for audio generation."](https://arxiv.org/abs/2406.19388) arXiv 2024.
8. Copet et al. ["Simple and controllable music generation."](https://arxiv.org/abs/2306.05284) NeurIPS 2023.
9. Tjandra et al. ["Meta Audiobox Aesthetics."](https://arxiv.org/abs/2502.05139) arXiv 2025.
10. Ma et al. ["CMI-RewardBench."](https://arxiv.org/abs/2603.00610) ICML 2026.
11. Kim et al. ["TuneJury."](https://arxiv.org/abs/2606.17006) arXiv 2026.
12. van den Oord et al. ["WaveNet: A generative model for raw audio."](https://arxiv.org/abs/1609.03499) arXiv 2016.
13. Défossez et al. ["High fidelity neural audio compression."](https://arxiv.org/abs/2210.13438) TMLR 2023.
14. Ho & Salimans. ["Classifier-free diffusion guidance."](https://arxiv.org/abs/2207.12598) arXiv 2022.
15. Sun et al. ["Fast best-of-N decoding via speculative rejection."](https://arxiv.org/abs/2410.20290) NeurIPS 2024.
16. Jamieson & Talwalkar. ["Non-stochastic best arm identification and hyperparameter optimization."](https://arxiv.org/abs/1502.07943) AISTATS 2016.
17. Kwon et al. ["Efficient memory management for LLM serving with PagedAttention."](https://arxiv.org/abs/2309.06180) SOSP 2023.
18. Chiang et al. ["Chatbot Arena."](https://arxiv.org/abs/2403.04132) ICML 2024.
19. Rafailov et al. ["Direct preference optimization."](https://arxiv.org/abs/2305.18290) NeurIPS 2023.
