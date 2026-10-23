# musicgen-prune-inference

**Best-of-N for text-to-music: sampling, judging and pruning under a compute budget.**
A long-form post with interactive figures, and its complete, reproducible companion code.

- **Read the post:** [r03922123.github.io/musicgen-prune-inference](https://r03922123.github.io/musicgen-prune-inference/)
  (source: [`post/post.md`](post/post.md); listed on [r03922123.github.io](https://r03922123.github.io/))
- **Check the evidence:** [`docs/report.md`](docs/report.md), the full study with every table, sample size and
  limitation; [`tasks/decisions.md`](tasks/decisions.md), every analysis decision with when it was made

Text-to-music models sample, so products generate several takes and keep the best one, which multiplies the cost
by N. The post builds the whole pipeline from the waveform up (codecs, MusicGen, guidance, sampling), surveys
how machines judge music (CLAP to 2026 reward models), measures where generation time actually goes, implements
early pruning so that it is exact, evaluates it, including a leak found in our own metric, and designs a
production best-of-N service.

## What is in the post, and where its code lives

| Section | Interactive figure | Code |
|---|---|---|
| From waveform to tokens | Fig. 1: three codebook layouts, and what each token can see | [`torch/prune.py`](torch/prune.py) `undelay` |
| The generator | | [`torch/prune.py`](torch/prune.py) `generate`, `sample` |
| Judging music | | [`torch/scorers.py`](torch/scorers.py), [`torch/agreement.py`](torch/agreement.py) |
| Best-of-N and its cost | Fig. 3 (and the lead figure): what N buys, by judge and selector, with bootstrap bands | [`torch/profile_decode.py`](torch/profile_decode.py) |
| Early pruning | Fig. 5: prune any prompt yourself, with its audio | [`torch/prune.py`](torch/prune.py), [`torch/evaluate.py`](torch/evaluate.py) `simulate` |
| A map of inference-time search | | post only |
| Evaluating a pruner | Fig. 6: the leak, as a rank scatter; Fig. 7: pruning scored on unheard audio | [`torch/frontier.py`](torch/frontier.py), [`torch/future.py`](torch/future.py) |
| System design | | post only |
| Humans in the loop | | [`torch/listen.py`](torch/listen.py), [`torch/bracket.py`](torch/bracket.py) |

The figures are not drawings. [`post/js/sim.js`](post/js/sim.js) is `torch/evaluate.py` ported to JavaScript; it
runs on [`post/data.json`](post/data.json) (the corpus scores) in the reader's browser, and
[`post/tests/sim.test.mjs`](post/tests/sim.test.mjs) fails CI if it stops reproducing `results/`.

## Run it yourself

The corpus (tokens and every score for 50 prompts × 8 takes) is committed, so levels 0 and 1 need no GPU.

```bash
# Level 0 — laptop, about a minute: the exactness self-check and every offline analysis
uv sync
uv run python torch/prune.py                 # pruning drops rows without changing survivors (+ a mutation that must fail)
uv run pytest torch/tests -q                 # evaluation, bracket, post data, and the post's numbers vs results/
uv run python torch/frontier.py              # schedules: fidelity (Q1) and value (Q2) tables + figure
uv run python torch/future.py                # the leak-free check (needs results/clap_chunks.json, committed)

# Level 1 — laptop with the audio (770 MB of WAVs from the corpus job, see docs/report.md)
uv run python torch/clap_chunks.py           # re-score every 10 s chunk with CLAP (CPU, ~25 min)
uv run python torch/listen.py                # blind A/B listening test

# Level 2 — GPU (Kaggle T4): regenerate the corpus, re-score judges, profile
#   kaggle_corpus/, kaggle_judge/, kaggle_cmi/, kaggle_profile/  (bundled by kaggle_job/bundle.py)

# The post
cd post && npm ci && node build.mjs          # -> post/dist/index.html (audio: see post/audio.lock)
```

Every push to `master` runs [`.github/workflows/site.yml`](.github/workflows/site.yml): Python tests, a check
that every number in the post matches `results/`, a check that the interactive figures recompute `results/`, the
build, a browser smoke test on desktop and mobile (figures respond, audio plays), then deploys to GitHub Pages.

## Repository map

| Path | Role |
|---|---|
| `post/` | the post (`post.md`), interactive figures (`js/`), their data, build script and tests |
| `torch/` | PyTorch + Hugging Face implementation: pruned generation, scorers, offline evaluation, profiler |
| `torch/tests/` | exactness, mutation and evaluation tests; `test_post_numbers.py` ties the post to `results/` |
| `results/` | every measured number, as committed JSON (and the frontier figure) |
| `data/` | prompts and the unpruned corpus (tokens + scores; WAVs are not committed) |
| `prune.py`, `bench.py`, `vendor/` | the original MLX implementation (Apple silicon) and its timing bench |
| `kaggle_*/` | GPU jobs |
| `docs/report.md` | the full research report |
| `SPEC.md`, `tasks/` | the project spec, plans and the decision log |

## Scope and honesty

One model (MusicGen-small), 30 s instrumental clips, 50 prompts, 8 takes each: the claims are about these and are
labelled *measured*, *derived* or *hypothesis* in the report. This is a personal project, not affiliated with any
company. MusicGen weights are CC-BY-NC 4.0 (non-commercial); MusicCaps captions are CC BY-SA 4.0.
