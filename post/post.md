---
date: October 24, 2026
author: r03922123
description: How text-to-music outputs are judged, from FID and CLIP score to FAD, CLAP and learned music judges; why picking the best of N takes by CLAP is a legitimate practice; and four ways to extend it, including pruning weak takes early. An interactive figure recomputes the pruning results from the data.
---

# Judging text-to-music: CLAP best-of-N, then and now

> **Summary.** A text-to-music model gives a different song every time, so a common recipe is to generate N takes
> and keep the one CLAP scores highest. This post traces how generated music is judged (precedent, past, present),
> explains why the CLAP recipe is legitimate and where it stops, and walks through four extensions. One of them,
> pruning weak takes early, is measured here on MusicGen.

<!-- toc -->

## Introduction

Type "lo-fi hip hop with warm piano" into [MusicGen](https://arxiv.org/abs/2306.05284) eight times and you get
eight different songs. Some are lovely; some wander off or go nearly silent. To return one good song, you need a way
to **score** a take automatically. When I used MusicGen, I generated N takes and kept the one with the highest
**CLAP score**. Two questions follow, and this post answers them:

1. **Is that legitimate?** Where do text-to-music metrics come from, and what do people use now?
2. **How can it be extended?** A better scorer, a cheaper search, a better generator, or people in the loop.

| Part | Question | Short answer |
|---|---|---|
| [Metrics](#how-generated-music-is-judged) | How is generated music judged? | Distribution metrics (FAD), text match (CLAP), people; now learned judges. |
| [CLAP best-of-N](#is-clap-best-of-n-legitimate) | Is picking by CLAP legitimate? | Yes, for text match: MusicGen reports CLAP, AudioLDM reranks with it. |
| [Extensions](#four-extensions) | Where next? | Better scorer, early pruning (measured), preference training, human tests. |

## How generated music is judged

Every metric below answers one of three questions: *does it sound real*, *does it match the text*, *do people like
it*. Two kinds of metric recur: **distribution metrics** compare a whole set of generated clips with a set of real
ones, and **per-clip metrics** score one clip, which is what you need to pick a take.

| Era | Metric | Question | Per clip? |
|---|---|---|---|
| Precedent (images) | FID | does the set look real | no |
| | CLIP score | does the image match the text | yes |
| Past (2023 music) | FAD | does the set sound real | no |
| | KL over audio tags | does it carry its reference recording's concepts | needs the reference |
| | CLAP score, MuLan score | does the audio match the text | yes |
| | Human ratings | quality and text relevance | yes |
| Present (2024–26) | FAD with music embeddings | does the set sound real | per song possible |
| | Audiobox Aesthetics | quality and enjoyment, as people rate it | yes |
| | Music reward models (CMI-RM) | which take people would prefer | yes |

### Precedent: images

**FID** ([Heusel et al. 2017](https://arxiv.org/abs/1706.08500)) embeds real and generated images with a pretrained
network, fits a Gaussian to each set, and measures the distance between the two Gaussians. Lower means the
generated set looks more like real data. It says nothing about any single image.

**CLIP score** ([Hessel et al. 2021](https://arxiv.org/abs/2104.08718)) scores one image against one caption.
[CLIP](https://arxiv.org/abs/2103.00020) (Radford et al. 2021) is two encoders, one for images and one for text,
trained so a picture and its caption land close together; the score is their cosine similarity. Because it works
per image, it can also *pick*: [DALL·E](https://arxiv.org/abs/2102.12092) (Ramesh et al. 2021) drew 512 images per
caption and kept the one CLIP ranked highest.

### Past: the MusicGen era

Audio copied both ideas. **FAD** ([Kilgour et al. 2019](https://arxiv.org/abs/1812.08466)) is FID on audio
embeddings (VGGish). **CLAP** ([Wu et al. 2022](https://arxiv.org/abs/2211.06687)) is CLIP for audio, and the
**CLAP score** is the cosine similarity between a clip and its prompt. MusicLM
([Agostinelli et al. 2023](https://arxiv.org/abs/2301.11325)) used its own text–music model, MuLan, for the same job.

MusicGen (Copet et al. 2023) set the standard report for text-to-music:

- **FAD** (VGGish): does the generated set sound like real music?
- **KL** over the labels of an audio-tagging classifier: does a clip carry the same concepts as its reference?
- **CLAP score**: does the clip match its text?
- **Human ratings** of overall quality (OVL) and relevance to the text (REL), from 1 to 100.

No single number covers everything, so papers report several, and people have the last word.

### Present: better embeddings and learned judges

Three trends since 2024:

- **FAD, done more carefully.** [Gui et al. (2024)](https://arxiv.org/abs/2311.01616) showed that FAD depends on the
  embedding, the reference set and the sample size, proposed a correction for small samples, and computed a
  per-song FAD that flags bad tracks.
- **Judges trained on human ratings.** [Audiobox Aesthetics](https://arxiv.org/abs/2502.05139) (Tjandra et al.
  2025) predicts people's 1–10 scores for production quality, production complexity, content enjoyment and content
  usefulness. It reads only the audio, not the prompt.
- **Music reward models.** [CMI-RM](https://arxiv.org/abs/2603.00610) (Ma et al. 2026) learns from people's choices
  between two takes and scores both musical quality and prompt match. Its authors use it to filter the top-$k$
  of many generations, the recipe of the next section with a stronger scorer.

## Is CLAP best-of-N legitimate?

**Best-of-N** means: draw $N$ samples, score each with a per-clip metric $r(x, y)$ (take $y$ for prompt $x$), keep
the highest. The LLM literature gives it this name, with a **verifier** or **reward model** as the scorer
([Cobbe et al. 2021](https://arxiv.org/abs/2110.14168); [WebGPT](https://arxiv.org/abs/2112.09332), Nakano et al.
2021).

Picking by CLAP stands on three facts:

1. **CLAP score is a standard text-to-music metric.** MusicGen reports it as *the* measure of text match.
2. **Reranking by a text–media model has precedent.** DALL·E reranked with CLIP. In audio,
   [AudioLDM](https://arxiv.org/abs/2301.12503) (Liu et al. 2023) does exactly this: its code takes
   `n_candidate_gen_per_text`, generates that many clips and keeps the one CLAP scores highest, and its reported
   metrics come from the best of 3. The Diffusers AudioLDM 2 pipeline offers the same as `num_waveforms_per_prompt`.
3. **CLAP works per clip and on short audio.** It needs no reference audio (FAD and KL do), and it can score a
   10 s prefix, which matters for [pruning](#2-a-cheaper-search-early-pruning).

**Where it stops.** CLAP measures *text match* only. It does not hear mixing, structure or enjoyment, which is why
MusicGen pairs it with FAD and human ratings. Two practical caveats:

- **Reported numbers.** If you publish a CLAP score for reranked outputs, say so: later work recomputed AudioLDM's
  metrics without reranking so that papers could be compared fairly
  ([Haji-Ali et al. 2024](https://arxiv.org/abs/2406.19388)).
- **Diminishing returns.** Roll one die and you expect 3.5. Keep the best of two: 4.47; of four: 5.24; of eight:
  5.72. Each doubling of $N$ buys less while the cost doubles. Music judges show the same shape: in TuneJury
  ([Kim et al. 2026](https://arxiv.org/abs/2606.17006)) the gain per doubling fell from +0.18–0.29 (4 → 8) to
  +0.06–0.14 (16 → 32).

**In practice:** CLAP best-of-N is a sound way to get the take that best matches the prompt. If you want "the take
people like best", you need a scorer that measures that.

## Four extensions

### 1. A better scorer

Replace or combine CLAP with a judge that hears more: add Audiobox Aesthetics for quality, or pick with a music
reward model such as CMI-RM, which was trained on people's choices. The search stays the same; only $r(x, y)$
changes.

### 2. A cheaper search: early pruning

Best-of-8 costs eight full songs, and seven of them are thrown away. MusicGen writes **left to right in time**, so
after 10 s each take has a prefix you can score. **Successive halving**, the schedule hyperparameter search uses to
stop bad training runs early ([Jamieson & Talwalkar 2016](https://arxiv.org/abs/1502.07943)), does the rest: start
8 takes, keep the best 4 by CLAP at 10 s, the best 2 at 20 s, finish those and pick the better.
Take-seconds: $8 \times 10 + 4 \times 10 + 2 \times 10 = 140$ against $8 \times 30 = 240$, so $0.583$ of the
compute. For LLMs, [Speculative Rejection](https://arxiv.org/abs/2410.20290) (Sun et al. 2024) does the same with a
reward model on partial answers.

The test is simple because CLAP is the reference: **does pruning keep the take that full best-of-8 would pick?**
I generated 8 takes for each of 50 prompts once, unpruned, saved CLAP scores of every 5 s prefix, and simulated each
schedule from those scores:

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
**Fig. 1. Prune it yourself.** That function, on the real CLAP scores of all 400 takes. Each line is one take's
score as it is generated. At each dashed checkpoint the lowest takes are cut (×); their faded continuation would
never be generated. ● is the pruned run's pick, ◎ unpruned best-of-8's. Press a take to hear it. **Try:** on
"orchestral build-up", keep-half cuts take D at 20 s, yet D is unpruned best-of-8's pick; keep 0.65 saves it. Then
set the first checkpoint to 5 s and watch the slow-build rate collapse.
:::

The prompts are 40 MusicCaps captions plus 10 hand-written **slow builds** (orchestral build-ups, post-rock
crescendos) whose best material comes late, a stress test for scoring early. Same-winner rate with 95% bootstrap
intervals over prompts ($n = 40$ and $n = 10$; `results/frontier.json`):

| Schedule (first checkpoint / keep) | Compute | Same-winner, MusicCaps | Same-winner, slow builds |
|---|---|---|---|
| 5 s / 0.5 | 0.354 | 0.62 [0.47, 0.78] | 0.30 [0.00, 0.60] |
| 10 s / 0.5 | 0.583 | 0.88 [0.78, 0.97] | 0.80 [0.50, 1.00] |
| **10 s / 0.65** | **0.750** | **0.97 [0.93, 1.00]** | **1.00** |
| 10 s / 0.75 | 0.792 | 1.00 | 1.00 |

Scoring at 5 s is too early, and the slow builds show it. From 10 s on, pruning keeps CLAP's pick at 0.75 of the
compute. The cutoff I used (same-winner ≥ 0.90) was chosen after seeing the results. Saved compute is not saved
time: on a T4 the 8 → 4 → 2 schedule cut generation from 38.6 s to 26.8 s (1.44×, against 1.71× fewer
take-seconds), because small-batch decoding is limited by kernel launches.
[Inside MusicGen decoding](decoding/) explains why, and how to prune without changing the survivors.

### 3. A better generator: train on preferences

Best-of-N fixes a bad sample after the fact. The alternative is to make good samples more likely: collect pairs
"take A preferred to take B" and fine-tune the generator on them with
[DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al. 2023), the method LLMs use. Then best-of-2 may do what
best-of-8 did.

### 4. People in the loop

Automatic metrics each see one side of "good", so in the end you ask people. The mainstream method is **pairwise
preference**: play two takes, ask which to keep listening to, and fit a **Bradley–Terry** model, in which a score
gap becomes a win probability. If take $a$ scores 1.2 and $b$ scores 0.4, $a$ wins $\sigma(0.8) \approx 69\%$ of the
time:

$$
P(a \text{ beats } b) = \sigma\big(r_a - r_b\big), \qquad \sigma(t) = \frac{1}{1 + e^{-t}} .
$$

[Chatbot Arena](https://arxiv.org/abs/2403.04132) (Chiang et al. 2024) ranks LLMs this way, and music reward models
learn from the same kind of choices. Match loudness, randomise order, and size the study first: 20 trials detect
only preferences of about 75% or more. The repo has such a test (`torch/listen.py`); it has not been run.

## Summary

- Generated music is judged by distribution metrics (FID → FAD), per-clip text match (CLIP score → CLAP score) and
  people; learned judges (Audiobox Aesthetics, music reward models) are the current addition.
- CLAP best-of-N is legitimate for *text match*: MusicGen reports CLAP, DALL·E reranked with CLIP, AudioLDM with
  CLAP. It does not measure quality or enjoyment.
- Extensions: a better scorer, early pruning (same pick on 97% of prompts at 0.75 of the compute), preference
  training, and pairwise human tests.

**Limitations.** One model (MusicGen-small), 30 s instrumental clips, 50 prompts, 8 takes each; slow-build
intervals are wide ($n = 10$). Only extension 2 is measured here.

## Reproducibility

`torch/tests/test_post_numbers.py` checks every table against the committed results, and `post/tests/sim.test.mjs`
checks the interactive figure; a mismatch fails the build. Every analysis decision, dated, is in the
[decision log](https://github.com/r03922123/musicgen-prune-inference/blob/master/tasks/decisions.md).

## Citation

> r03922123. (Oct 2026). Judging text-to-music: CLAP best-of-N, then and now.
> https://r03922123.github.io/musicgen-prune-inference/

```text
@article{r03922123_2026_judging_music,
  title   = "Judging text-to-music: CLAP best-of-N, then and now",
  author  = "r03922123",
  journal = "r03922123.github.io",
  year    = "2026",
  month   = "Oct",
  url     = "https://r03922123.github.io/musicgen-prune-inference/"
}
```

## References

1. Heusel et al. ["GANs trained by a two time-scale update rule converge to a local Nash equilibrium."](https://arxiv.org/abs/1706.08500) NeurIPS 2017. (FID)
2. Hessel et al. ["CLIPScore: A reference-free evaluation metric for image captioning."](https://arxiv.org/abs/2104.08718) EMNLP 2021.
3. Radford et al. ["Learning transferable visual models from natural language supervision."](https://arxiv.org/abs/2103.00020) ICML 2021.
4. Ramesh et al. ["Zero-shot text-to-image generation."](https://arxiv.org/abs/2102.12092) ICML 2021.
5. Kilgour et al. ["Fréchet Audio Distance: A metric for evaluating music enhancement algorithms."](https://arxiv.org/abs/1812.08466) Interspeech 2019.
6. Wu et al. ["Large-scale contrastive language-audio pretraining."](https://arxiv.org/abs/2211.06687) ICASSP 2023.
7. Agostinelli et al. ["MusicLM: Generating music from text."](https://arxiv.org/abs/2301.11325) arXiv 2023.
8. Copet et al. ["Simple and controllable music generation."](https://arxiv.org/abs/2306.05284) NeurIPS 2023.
9. Gui et al. ["Adapting Fréchet Audio Distance for generative music evaluation."](https://arxiv.org/abs/2311.01616) ICASSP 2024.
10. Tjandra et al. ["Meta Audiobox Aesthetics."](https://arxiv.org/abs/2502.05139) arXiv 2025.
11. Ma et al. ["CMI-RewardBench."](https://arxiv.org/abs/2603.00610) ICML 2026.
12. Cobbe et al. ["Training verifiers to solve math word problems."](https://arxiv.org/abs/2110.14168) arXiv 2021.
13. Nakano et al. ["WebGPT: Browser-assisted question-answering with human feedback."](https://arxiv.org/abs/2112.09332) arXiv 2021.
14. Liu et al. ["AudioLDM: Text-to-audio generation with latent diffusion models."](https://arxiv.org/abs/2301.12503) ICML 2023.
15. Haji-Ali et al. ["Taming data and transformers for audio generation."](https://arxiv.org/abs/2406.19388) arXiv 2024.
16. Kim et al. ["TuneJury."](https://arxiv.org/abs/2606.17006) arXiv 2026.
17. Jamieson & Talwalkar. ["Non-stochastic best arm identification and hyperparameter optimization."](https://arxiv.org/abs/1502.07943) AISTATS 2016.
18. Sun et al. ["Fast best-of-N decoding via speculative rejection."](https://arxiv.org/abs/2410.20290) NeurIPS 2024.
19. Rafailov et al. ["Direct preference optimization."](https://arxiv.org/abs/2305.18290) NeurIPS 2023.
20. Chiang et al. ["Chatbot Arena."](https://arxiv.org/abs/2403.04132) ICML 2024.
