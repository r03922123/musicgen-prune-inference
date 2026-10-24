# musicgen-prune-inference

**Best-of-N for text-to-music: sampling, judging and pruning under a compute budget.**
A post with interactive figures, and the complete code and data behind every number in it.

- **Read the post:** [r03922123.github.io/musicgen-prune-inference](https://r03922123.github.io/musicgen-prune-inference/)
  (source: [`post/post.md`](post/post.md))
- **Every analysis decision, dated:** [`tasks/decisions.md`](tasks/decisions.md)

Text-to-music models sample, so products generate several takes and keep the best, at N times the cost. The post
builds MusicGen from the waveform up, picks with one automatic judge and grades with others (the standard guard
against reward hacking), measures where decoding time goes, makes early pruning exact, and evaluates it, including
a leak found in its own metric.

## Where each part of the post lives

| Post section | Interactive figure | Code |
|---|---|---|
| From waveform to tokens | Fig. 1: codebook layouts | [`torch/prune.py`](torch/prune.py) `undelay` |
| The generator | | [`torch/prune.py`](torch/prune.py) `generate`, `sample` |
| Best-of-N and its judges | Fig. 3: pick with CLAP, grade with someone else | [`torch/scorers.py`](torch/scorers.py), [`torch/agreement.py`](torch/agreement.py) |
| What decoding costs | | [`torch/profile_decode.py`](torch/profile_decode.py) |
| Early pruning | Fig. 5: prune any prompt, with audio | [`torch/evaluate.py`](torch/evaluate.py) `simulate` |
| Evaluating a pruner | Figs. 6–7: the leak, and the leak-free check | [`torch/frontier.py`](torch/frontier.py), [`torch/future.py`](torch/future.py) |
| Humans | | [`torch/listen.py`](torch/listen.py), [`torch/bracket.py`](torch/bracket.py) |

The figures are not drawings: [`post/js/sim.js`](post/js/sim.js) is `torch/evaluate.py` ported to JavaScript, run
on [`post/data.json`](post/data.json) in the reader's browser, and [`post/tests/sim.test.mjs`](post/tests/sim.test.mjs)
fails CI if it stops reproducing `results/`.

## Run it yourself

The corpus (tokens and every score, 50 prompts × 8 takes) is committed, so the first block needs no GPU.

```bash
uv sync
uv run python torch/prune.py                 # exactness self-check, plus mutations that must fail
uv run pytest torch/tests -q                 # evaluation tests, and every number in the post vs results/
uv run python torch/frontier.py              # Q1 / Q2 tables
uv run python torch/future.py                # the leak-free check
uv run python torch/agreement.py             # judge agreement

cd post && npm ci && node build.mjs          # the post -> post/dist/index.html
```

With the audio (400 WAVs, 770 MB, not committed): fetch them with
`uvx kaggle kernels output samlinlinlin/musicgen-prune-corpus -p data/wavs_dl --file-pattern 'wavs\.tar'` and untar
into `data/wavs/`. Then `torch/clap_chunks.py` re-scores every 10 s chunk (CPU, ~25 min), `torch/listen.py` runs a
blind A/B test, and `torch/build_post_data.py` rebuilds the post's data and MP3s (the published MP3s are the release
asset pinned in `post/audio.lock`).

GPU jobs run on a Kaggle T4, one folder per kernel (`kaggle_corpus/`, `kaggle_judge/`, `kaggle_cmi/`,
`kaggle_profile/`, `kaggle_job/`; sources are embedded by `kaggle_job/bundle.py`). The original MLX implementation
for Apple silicon is `prune.py` and `bench.py` at the root (`uv run python bench.py steps` wrote
`results_steps_small_float16.json`).

Every push to `master` runs [`.github/workflows/site.yml`](.github/workflows/site.yml): Python tests, the post's
numbers against `results/`, the figures against `results/`, the build, a browser smoke test on desktop and mobile,
then deploy to GitHub Pages.

## Scope

One model (MusicGen-small), 30 s instrumental clips, 50 prompts, 8 takes each; the claims are about these. A
personal project, not affiliated with any company. MusicGen weights are CC-BY-NC 4.0 (non-commercial); MusicCaps
captions are CC BY-SA 4.0.
