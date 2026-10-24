---
date: October 24, 2026
author: r03922123
description: How a text-to-music model turns text into audio tokens, how to pick the best of N takes without fooling yourself, where decoding time goes, and how to stop weak takes early, exactly and measurably. Interactive figures recompute every result from the data.
---

# Best-of-N for text-to-music: sampling, judging and pruning under a compute budget

> **Summary.** Text-to-music models sample, so products generate several takes and keep the best, at N times the
> cost. This post builds the pipeline from the waveform up, picks takes the way LLM evaluation does, measures where
> decoding time goes, and stops weak takes early without changing the winner.

:::figure bestofn compact=1
**The result in one picture.** A prompt's eight takes really differ: picking with the reward model CMI-RM itself
(green) gains a lot over a random take. CLAP, the cheap judge this study selects with, recovers about a quarter of
that (blue). Pruning can make CLAP's choice cheaper, not better. The bottleneck is the judge.
:::

<!-- toc -->

## Introduction

Type "lo-fi hip hop with warm piano" into a text-to-music model eight times and you get eight different songs. Some
are lovely; some wander off or go nearly silent. The model defines a *distribution* over songs, and every call draws
one sample. So products do what LLM products do: generate several candidates, score them, keep the best. This is
**best-of-N**, and it multiplies the cost of every request by N. Can you tell early which takes are worth
finishing?

The study behind this post is small and reproducible: MusicGen-small, 50 prompts, 8 takes each, a Kaggle T4 GPU and
a laptop. Every number comes from a committed results file in the
[companion repo](https://github.com/r03922123/musicgen-prune-inference), or is derived, and I say which.^[Five
minutes? Skip to [Fig. 5](#early-pruning), pick "orchestral build-up", press play on take D, then move the first
checkpoint from 10 s to 5 s.]

| Part | Question | Short answer |
|---|---|---|
| [Tokens](#from-waveform-to-tokens), [generator](#the-generator) | What is generated? | 6,000 tokens per 30 s take, drawn with random noise. |
| [Judges](#best-of-n-and-its-judges) | Who picks the best take? | A learned judge; the cheap one picks barely better than chance. |
| [Cost](#what-decoding-costs) | Where does the time go? | N times the work, and the GPU mostly waits. |
| [Pruning](#early-pruning), [evaluation](#evaluating-a-pruner) | Can we stop bad takes early? | Yes, exactly; but cheaper is not better. |

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
generator. Exact pruning depends on this. Note also that MusicGen writes **left to right in time**, 20 ms of audio
per step, so a 10 s prefix can be judged while the rest is unwritten.

## Best-of-N and its judges

Best-of-N has two halves: draw $N$ takes, then pick one. Picking needs a **judge**, a function $r(x, y)$ that scores
one take $y$ of prompt $x$, higher meaning better. LLM papers call it a *reward model* or a *verifier*.

### What does N buy?

Roll one die and you expect 3.5. Keep the best of two: 4.47. Of four: 5.24. Of eight: 5.72. Each doubling buys less
than the last, while cost doubles every time.^[For normally distributed scores the best of $N$ grows like
$\sqrt{2 \ln N}$ standard deviations.] Music reward models show the same shape: in TuneJury
([Kim et al. 2026](https://arxiv.org/abs/2606.17006)) the gain per doubling fell from +0.18–0.29 (4 → 8) to
+0.06–0.14 (16 → 32). And that assumes a perfect judge.

### The judges in this study

"Better" mixes prompt-following, sound quality and enjoyment, and no judge measures all three. The study uses four:

| Judge | Reads the prompt? | Learned from | Measures |
|---|---|---|---|
| CLAP | yes | audio–caption pairs | does the audio match the words |
| MuQ-MuLan | yes | music–text pairs | the same, music-specific |
| Audiobox Aesthetics | no | human 1–10 ratings | quality and enjoyment |
| CMI-RM | yes | human preferences between two takes | quality and prompt match, as people judge them |

**Audio–text judges.** [CLAP](https://arxiv.org/abs/2211.06687) (Wu et al. 2022) is CLIP for audio: two encoders
trained so a clip and its caption land close together; the score is their cosine similarity.
[MuQ-MuLan](https://arxiv.org/abs/2501.01108) (2025) does the same on music only. One trap: CLAP sees at most **10
seconds** and randomly crops anything longer, so I score each 30 s clip as three 10 s chunks and average:

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

Remember: **a 30 s CLAP score averages three 10 s chunks.** It comes back later.

**Judges trained on people.** [Audiobox Aesthetics](https://arxiv.org/abs/2502.05139) (Tjandra et al. 2025)
regresses human 1–10 ratings; I use its *content enjoyment* (CE) score. **Reward models** learn from comparisons
instead, because people are unreliable about absolute scores but reliable about "which of these two?":

1. A network gives every take a score $r(x, y)$.
2. A score *gap* becomes a probability of being preferred: if take $a$ scores 1.2 and $b$ scores 0.4, $a$ wins
   $\sigma(0.8) \approx 69\%$ of the time, where $\sigma(t) = 1/(1 + e^{-t})$.
3. Training raises the probability of the takes people actually preferred.

That is the **Bradley–Terry model**, the same one behind LLM reward models:

$$
P(a \text{ beats } b) = \sigma\big(r(x, y_a) - r(x, y_b)\big), \qquad
\mathcal{L} = -\log \sigma\big(r(x, y_{\text{win}}) - r(x, y_{\text{lose}})\big) .
$$

[CMI-RM](https://arxiv.org/abs/2603.00610) (ICML 2026) is one, with a head for musical quality and one for prompt
match.

### Pick with one judge, grade with another

A judge has blind spots. Pick with a judge and grade with the *same* judge, and you measure how well you found its
blind spots. LLM alignment calls this **reward hacking**: [Gao et al. (2023)](https://arxiv.org/abs/2210.10760) ran
best-of-$n$ against a *proxy* reward model while a separate *gold* model graded the picks, and the gold score rose
far less than the proxy's, eventually falling. The standard remedy: **select with one judge, grade with judges the
selector never saw.**

Here CLAP selects, because it is cheap and can score a 10 s prefix, which pruning needs. The other three grade. The
y-axis is the **gain over a random take**: the grader's score of the pick minus the average of all 8 takes. The
**oracle** picks with the grader itself, the most these takes could give.

:::figure bestofn
**Fig. 3. Pick with CLAP, grade with someone else.** For each $N$: the gain, averaged over MusicCaps prompts and
every subset of $N$ of the 8 takes, with 95% bootstrap bands over prompts. **Try:** switch the grader. CLAP stays far
below the oracle under every grader, and under MuQ-MuLan, built for CLAP's own job, it is flat at zero.
**Caveat:** subsets come from a fixed pool of 8, so this says nothing about $N > 8$.
:::

Why the gap? Out of a prompt's 8 takes, how often do two judges pick the same favourite? At random, 1 time in 8
(0.125). CLAP and MuQ-MuLan agree on 0.16 [0.06, 0.26] of prompts, CLAP and CMI-RM quality on 0.18 [0.08, 0.30]
($n = 50$, `results/judge_agreement.json`).^[They do agree on which *prompts* went well, because some prompts are
easier: averaged per prompt, CLAP and MuQ-MuLan correlate at +0.35. Best-of-N never compares takes of different
prompts, so that does not help.] Barely above chance.

**In practice:** say *whose* best you mean, and report results under graders the selector never saw.

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

![GPU timeline](figures/timeline.svg "**Fig. 4. Where a decoding step goes (schematic).** The CPU spends the step launching about 500 tiny kernels through Python and PyTorch; the GPU finishes each in microseconds and then waits. Measured on a T4 at batch 1: 18.7 ms per step, of which the GPU is busy 6.5 ms.")

Python launches about 500 tiny kernels per step, more slowly than the GPU runs them, so the GPU idles 12.2 of every
18.7 ms. Small-batch eager decoding is **launch-bound**. So fewer rows barely help, and **pruning saves less time
than compute.** The standard fix is a **CUDA graph**: record a step once, replay it with one launch. Graphs need
fixed shapes, which matters for pruning; I have not measured one here.

## Early pruning

Most of best-of-N's compute goes into takes that lose. Can we stop them partway without changing which take wins?
This is early pruning, a cousin of beam search; for LLMs, [Speculative Rejection](https://arxiv.org/abs/2410.20290)
(Sun et al. 2024) does the same with a reward model on partial answers. It can only be as good as a judge that
works on prefixes.

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
**Fig. 5. Prune it yourself.** That function, on the real CLAP scores of all 400 takes. Each line is one take's
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
per schedule becomes a millisecond loop; that is why Fig. 5 runs in your browser. Measured for 8 → 4 → 2
(generation only; M1: `results_steps_small_float16.json`, T4: the step-time file above):

| Hardware | Compute saved | Predicted from step times | Measured |
|---|---|---|---|
| Apple M1, MLX | 1.71× | 1.48× | 1.59× (104 s → 65 s) |
| Tesla T4, PyTorch eager | 1.71× | 1.44× | 1.44× (38.6 s → 26.8 s) |

The T4 falls short because of the launch-bound floor. **Caveat:** these use a cheap stand-in scorer; a real judge
at every checkpoint costs extra.

## Evaluating a pruner

Exact pruning leaves survivors unchanged, but it can still cut the take that would have won. The test set is 40
MusicCaps captions plus 10 hand-written **slow-build** prompts (orchestral build-ups, post-rock crescendos) that
save their best material for late: a stress test on which judging early should fail. Every schedule is simulated
from the one unpruned corpus, with 95% bootstrap intervals over prompts. Ask two questions:

- **Q1, fidelity:** does pruning keep the take unpruned best-of-8 would pick (the *same-winner rate*)? It grades
  the pruner.
- **Q2, quality:** is that pick better than a random take, under a grader the selector never saw? It grades the
  selector.

A pruner can ace Q1 and still be worthless, if what it faithfully keeps is no better than chance.

Q1 ($n = 40$ MusicCaps, $n = 10$ slow builds; `results/frontier.json`):

| Schedule (first checkpoint / keep) | Compute | Same-winner, MusicCaps | Same-winner, slow builds |
|---|---|---|---|
| 5 s / 0.5 | 0.354 | 0.62 [0.47, 0.78] | 0.30 [0.00, 0.60] |
| 10 s / 0.5 | 0.583 | 0.88 [0.78, 0.97] | 0.80 [0.50, 1.00] |
| **10 s / 0.65** | **0.750** | **0.97 [0.93, 1.00]** | **1.00** |
| 10 s / 0.75 | 0.792 | 1.00 | 1.00 |

Judging at 5 s is too early; the slow builds show it. At 10 s most of the problem goes away.

Q2, Fig. 3 at $N = 8$ (MusicCaps, $n = 40$; `results/frontier.json`, `selection`):

| Grader | Gain of CLAP's pick vs a random take | Oracle gain |
|---|---|---|
| CMI-RM quality | +0.157 [+0.010, +0.300] | 0.590 |
| CMI-RM alignment | +0.110 [−0.004, +0.219] | 0.432 |
| Audiobox CE | +0.193 [−0.040, +0.422] | 1.115 |
| MuQ-MuLan | −0.008 [−0.030, +0.015] | 0.111 |

Only one interval excludes zero, and not after a Bonferroni correction for four tests ([−0.029, +0.340]).
**Pruning cheaply preserves a choice that is barely better than chance.**

### A leak in my own metric

The 0.97 looks great, but it suffers from **target leakage**: the predictor is part of what it predicts. Think of
predicting final grades from the midterm when the final grade *includes* the midterm. Here a 30 s CLAP score
averages three 10 s chunks, and the 10 s checkpoint score *is*, up to decoding noise, the first one:

:::figure leak
**Fig. 6. The leak, made visible.** Each dot is one take, placed by its rank within its prompt: by its 10 s score
(x) and by a target (y). Against the chunk it has heard, ranks sit on the diagonal (rank correlation ρ = +0.99;
+1 is the same order, 0 unrelated). Against the 20 s it has not heard, they spread out.
:::

The standard fix: **define the target on data the predictor has not seen.** Before computing anything, I committed
this check: the target is the take CLAP likes best on the **last** 10 s. If $k$ of 8 takes survive at random, it
survives with probability $k/8$ (`results/future_only.json`):

| Schedule | MusicCaps: kept vs chance | Excess [95% CI] | Slow-build excess |
|---|---|---|---|
| 10 s / 0.5 | 0.72 vs 0.25 | +0.47 [+0.33, +0.60] | +0.25 [−0.05, +0.55] |
| 10 s / 0.65 | 0.88 vs 0.50 | +0.38 [+0.28, +0.47] | +0.30 [+0.00, +0.50] |
| 10 s / 0.75 | 0.93 vs 0.62 | +0.30 [+0.20, +0.38] | +0.17 [−0.12, +0.38] |

The early signal is real, but weaker: measured cleanly, the 10 s score correlates with the unheard 10–30 s at
$\rho = +0.44$, not $+0.99$, and on slow builds at only +0.25 [−0.01, +0.48]. Two lessons that carry beyond music:
define targets the predictor has not seen, and **write the analysis down before running it**. My Q1 threshold
(same-winner ≥ 0.90) was chosen after seeing results, and I say so; the leak check was committed first.

## System design: a best-of-N music service

A service takes a prompt and returns one song, running best-of-N with pruning behind the API. Each decision maps to
a measurement above.

![Service architecture](figures/system.svg "**Fig. 7. A best-of-N music service.** The decode loop shares GPUs through continuous batching; a checkpoint scorer decides what to drop; a different model picks the keeper; an offline loop replays logged requests to evaluate new scorers and schedules exactly.")

**Budget first.** On one T4, best-of-8 took 38.6 s of GPU time, pruned 26.8 s, one take about 27 s. The real
choice is **best-of-8 with pruning vs best-of-2 without**, and Fig. 3 says what the extra takes buy.

- **Release slots, do not shrink batches** (launch-bound). A pruned take's slot goes to another request at once
  (continuous batching, as in [vLLM](https://arxiv.org/abs/2309.06180)): pruning pays off as **throughput**.
- **Page the KV cache** (2.2 GiB). The cache limits how many takes fit; paging frees a pruned take's memory at once.
- **One CUDA graph per batch size** (the GPU idles 65% of each step), since pruning changes the batch.
- **Count the scorer** and overlap it with decoding on its own stream.
- **Pick the early scorer for agreement with the final grader**: CLAP picks the reward model's favourite on only
  18% of prompts.
- **Replay offline** (pruning is exact): log tokens and scores, and test new schedules on real traffic.

## Humans and what comes next

Machine judges disagree, so eventually you need people. The mainstream method is **pairwise preference**: show two
takes, ask which to keep listening to, and fit Bradley–Terry, as
[Chatbot Arena](https://arxiv.org/abs/2403.04132) (Chiang et al. 2024) does for LLMs. Match loudness, randomise
order, and size the study first: 20 trials detect only preferences of about 75% or more. The repo has such a test
(`torch/listen.py`); it has not been run. Next steps: train the generator on preferences
([DPO](https://arxiv.org/abs/2305.18290)) so one take is good more often, and train early scorers to predict the
final grader rather than CLAP.

## Summary

- Best-of-N costs N times as much, and each doubling of N buys less.
- A codec turns 960,000 samples into 6,000 tokens; guidance doubles the batch; a take is a function of its prompt
  and its own noise.
- Pick with one judge, grade with others: CLAP matches another judge's favourite on 16–18% of prompts (chance:
  12.5%).
- Small-batch decoding is launch-bound (GPU busy 35%), which caps what pruning saves.
- Noise keyed by take makes pruning exact, so one unpruned corpus evaluates every schedule.
- Ask fidelity and quality; keep targets unseen by the predictor; write the analysis down first.

**Limitations.** One model, 30 s instrumental clips, 50 prompts, 8 takes, one seed each; slow-build intervals are
wide ($n = 10$). CMI-RM builds on MuQ, so those judges are not fully independent. No human result yet.

## Exercises

1. **Codec arithmetic.** 9 codebooks of 1024 entries at 86 frames per second: what bitrate, and how many tokens
   for a 3-minute song?
2. **KV cache.** Write its size for $R$ rows, $T$ frames, $L$ layers, width $d$, fp16. How many 30 s takes with
   guidance fit in 16 GB beside MusicGen-small?
3. **Exactness.** In `torch/prune.py`, set `row_keyed=False` in the pruned call and explain the token-match number.

## Reproducibility

`torch/tests/test_post_numbers.py` checks every table against the committed results, and `post/tests/sim.test.mjs`
checks every interactive figure; a mismatch fails the build. Every analysis decision, dated, is in the
[decision log](https://github.com/r03922123/musicgen-prune-inference/blob/master/tasks/decisions.md).

## Citation

> r03922123. (Oct 2026). Best-of-N for text-to-music: sampling, judging and pruning under a compute budget.
> https://r03922123.github.io/musicgen-prune-inference/

```text
@article{r03922123_2026_bestofn_music,
  title   = "Best-of-N for text-to-music: sampling, judging and pruning under a compute budget",
  author  = "r03922123",
  journal = "r03922123.github.io",
  year    = "2026",
  month   = "Oct",
  url     = "https://r03922123.github.io/musicgen-prune-inference/"
}
```

## References

1. van den Oord et al. ["WaveNet: A generative model for raw audio."](https://arxiv.org/abs/1609.03499) arXiv 2016.
2. Défossez et al. ["High fidelity neural audio compression."](https://arxiv.org/abs/2210.13438) TMLR 2023.
3. Copet et al. ["Simple and controllable music generation."](https://arxiv.org/abs/2306.05284) NeurIPS 2023.
4. Ho & Salimans. ["Classifier-free diffusion guidance."](https://arxiv.org/abs/2207.12598) arXiv 2022.
5. Wu et al. ["Large-scale contrastive language-audio pretraining."](https://arxiv.org/abs/2211.06687) ICASSP 2023.
6. Zhu et al. ["MuQ: Self-supervised music representation learning."](https://arxiv.org/abs/2501.01108) arXiv 2025.
7. Tjandra et al. ["Meta Audiobox Aesthetics."](https://arxiv.org/abs/2502.05139) arXiv 2025.
8. Ma et al. ["CMI-RewardBench."](https://arxiv.org/abs/2603.00610) ICML 2026.
9. Kim et al. ["TuneJury."](https://arxiv.org/abs/2606.17006) arXiv 2026.
10. Gao, Schulman & Hilton. ["Scaling laws for reward model overoptimization."](https://arxiv.org/abs/2210.10760) ICML 2023.
11. Sun et al. ["Fast best-of-N decoding via speculative rejection."](https://arxiv.org/abs/2410.20290) NeurIPS 2024.
12. Jamieson & Talwalkar. ["Non-stochastic best arm identification and hyperparameter optimization."](https://arxiv.org/abs/1502.07943) AISTATS 2016.
13. Kwon et al. ["Efficient memory management for LLM serving with PagedAttention."](https://arxiv.org/abs/2309.06180) SOSP 2023.
14. Chiang et al. ["Chatbot Arena."](https://arxiv.org/abs/2403.04132) ICML 2024.
15. Rafailov et al. ["Direct preference optimization."](https://arxiv.org/abs/2305.18290) NeurIPS 2023.
