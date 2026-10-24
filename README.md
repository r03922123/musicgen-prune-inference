# musicgen-prune-inference

Two posts with interactive figures, and the complete code and data behind every number in them.

1. **[Judging text-to-music: CLAP best-of-N, then and now](https://r03922123.github.io/musicgen-prune-inference/)**
   ([`post/post.md`](post/post.md), 10 min). How generated music is judged (FID and CLIP score → FAD and CLAP →
   learned judges), why picking the best of N takes by CLAP is legitimate, and four extensions. One is measured:
   pruning weak takes early keeps CLAP's pick on 97% of prompts at 0.75 of the compute.
2. **[Inside MusicGen decoding: tokens, guidance and exact pruning](https://r03922123.github.io/musicgen-prune-inference/decoding/)**
   ([`post/decoding.md`](post/decoding.md), 8 min). RVQ and the delay pattern, guidance and Gumbel-max sampling,
   where a decoding step's time goes on a T4, pruning a running batch without changing the survivors, and the design
   of a best-of-N service.

Every analysis decision, dated: [`tasks/decisions.md`](tasks/decisions.md).

## Where each part lives

| Post, section | Interactive figure | Code |
|---|---|---|
| 1, How generated music is judged | Fig. 1: metrics timeline (static) | |
| 1, Is CLAP best-of-N legitimate? | | [`torch/scorers.py`](torch/scorers.py) (CLAP and the other metrics) |
| 1, A cheaper search: early pruning | Fig. 2: prune any prompt, with audio | [`torch/evaluate.py`](torch/evaluate.py) `simulate`, [`torch/frontier.py`](torch/frontier.py) |
| 1, People in the loop | | [`torch/listen.py`](torch/listen.py) (pairwise A/B test, not yet run) |
| 2, From waveform to tokens | Fig. 1: codebook layouts | [`torch/prune.py`](torch/prune.py) `undelay` |
| 2, The generator | | [`torch/prune.py`](torch/prune.py) `generate`, `sample` |
| 2, What decoding costs | | [`torch/profile_decode.py`](torch/profile_decode.py) |
| 2, Making pruning exact | | [`torch/prune.py`](torch/prune.py) (self-check with mutations) |

The interactive figures are not drawings: [`post/js/sim.js`](post/js/sim.js) is `torch/evaluate.py` ported to JavaScript, run
on [`post/data.json`](post/data.json) in the reader's browser, and [`post/tests/sim.test.mjs`](post/tests/sim.test.mjs)
fails CI if it stops reproducing `results/`.

## Run it yourself

The corpus (tokens and every score, 50 prompts × 8 takes) is committed, so the first block needs no GPU.

```bash
uv sync
uv run python torch/prune.py                 # exactness self-check, plus mutations that must fail
uv run pytest torch/tests -q                 # evaluation tests, and every number in the post vs results/
uv run python torch/frontier.py              # the pruning table

cd post && npm ci && node build.mjs          # both posts -> post/dist/
```

With the audio (400 WAVs, 770 MB, not committed): fetch them with
`uvx kaggle kernels output samlinlinlin/musicgen-prune-corpus -p data/wavs_dl --file-pattern 'wavs\.tar'` and untar
into `data/wavs/`. Then `torch/listen.py` runs a blind A/B test, and `torch/build_post_data.py` rebuilds the post's
data and MP3s (the published MP3s are the release asset pinned in `post/audio.lock`).

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
