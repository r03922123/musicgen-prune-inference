# musicgen-prune-inference

**Best-of-N text-to-music: sampling, judging and pruning under a compute budget.**
A single graduate-level lecture with its complete, reproducible companion code.

- **Read the lecture:** [r03922123.github.io/musicgen-prune-inference](https://r03922123.github.io/musicgen-prune-inference/)
  (source: [`lecture/lecture.md`](lecture/lecture.md), which GitHub also renders)
- **Play the demo:** [Eight Takes](https://r03922123.github.io/musicgen-prune-inference/play/), knock out
  eight AI takes of one prompt on 10, 20 and 30 s clips, against an AI picker
- **Check the evidence:** [`docs/report.md`](docs/report.md), the full study with every table, sample size and
  limitation; [`tasks/decisions.md`](tasks/decisions.md), every analysis decision with when it was made

Text-to-music models sample, so products generate several takes and keep the best one, which multiplies the cost
by N. The lecture builds the whole pipeline from the waveform up (codecs, MusicGen, guidance, sampling), looks at
how machines judge music (CLAP to 2026 reward models), measures where generation time actually goes, implements
early pruning so that it is exact, evaluates it, including a leak found in our own metric, and designs a
production best-of-N service.

## Syllabus

| # | Section | What you will be able to do | Code |
|---|---|---|---|
| 1 | Introduction | state best-of-N and its cost | |
| 2 | From waveform to tokens | compute codec rates and token counts; implement RVQ; read the delay pattern | [`torch/prune.py`](torch/prune.py) `undelay` |
| 3 | The generator | count MusicGen's parameters; implement CFG and top-k as Gumbel-max | [`torch/prune.py`](torch/prune.py) `generate`, `sample` |
| 4 | Judging music | explain FAD vs CLAP vs reward models; measure judge agreement | [`torch/scorers.py`](torch/scorers.py), [`torch/agreement.py`](torch/agreement.py) |
| 5 | Best-of-N and its cost | estimate a decode step's lower bound; read a profiler trace | [`torch/profile_decode.py`](torch/profile_decode.py) |
| 6 | Early pruning | implement successive halving; make it exact (row-keyed noise, cache slicing) | [`torch/prune.py`](torch/prune.py), [`torch/evaluate.py`](torch/evaluate.py) `simulate` |
| 7 | Evaluating a pruner | separate fidelity from value; find and remove a target leak | [`torch/frontier.py`](torch/frontier.py), [`torch/future.py`](torch/future.py) |
| 8 | System design | design a best-of-N service and its real-time variant | lecture only |
| 9 | Humans in the loop | design a pairwise listening study; analyse brackets | [`torch/listen.py`](torch/listen.py), [`torch/bracket.py`](torch/bracket.py), [`site/`](site/) |
| 10 | Where the field is going | place steering, better early scorers and real time | lecture only |

The lecture ends with eight exercises.

## Run it yourself

The corpus (tokens and every score for 50 prompts × 8 takes) is committed, so levels 0 and 1 need no GPU.

```bash
# Level 0 — laptop, about a minute: the exactness self-check and every offline analysis
uv sync
uv run python torch/prune.py                 # pruning drops rows without changing survivors (+ a mutation that must fail)
uv run pytest torch/tests -q                 # evaluation, bracket, site data, and the lecture's numbers vs results/
uv run python torch/frontier.py              # schedules: fidelity (Q1) and value (Q2) tables + figure
uv run python torch/future.py                # the leak-free check (needs results/clap_chunks.json, committed)

# Level 1 — laptop with the audio (770 MB of WAVs from the corpus job, see docs/report.md)
uv run python torch/clap_chunks.py           # re-score every 10 s chunk with CLAP (CPU, ~25 min)
uv run python torch/listen.py                # blind A/B listening test

# Level 2 — GPU (Kaggle T4): regenerate the corpus, re-score judges, profile
#   kaggle_corpus/, kaggle_judge/, kaggle_cmi/, kaggle_profile/  (bundled by kaggle_job/bundle.py)

# The lecture and the game
cd lecture && npm ci && node build.mjs       # -> lecture/dist/index.html
```

Every push to `master` runs [`.github/workflows/site.yml`](.github/workflows/site.yml): Python and JS tests,
a check that every headline number in the lecture matches `results/`, the lecture build, a browser smoke test of the
lecture and of a full game, then deploys to GitHub Pages.

## Repository map

| Path | Role |
|---|---|
| `lecture/` | the lecture (`lecture.md`), figures, build script and smoke test |
| `torch/` | PyTorch + Hugging Face implementation: pruned generation, scorers, offline evaluation, profiler |
| `torch/tests/` | exactness, mutation and evaluation tests; `test_lecture_numbers.py` ties the lecture to `results/` |
| `results/` | every measured number, as committed JSON (and the frontier figure) |
| `data/` | prompts and the unpruned corpus (tokens + scores; WAVs are not committed) |
| `site/` | the Eight Takes game (served at `/play/`) |
| `prune.py`, `bench.py`, `vendor/` | the original MLX implementation (Apple silicon) and its timing bench |
| `kaggle_*/` | GPU jobs |
| `docs/report.md` | the full research report |
| `SPEC.md`, `tasks/` | the project spec, plans and the decision log |

## Scope and honesty

One model (MusicGen-small), 30 s instrumental clips, 50 prompts, 8 takes each: the claims are about these and are
labelled *measured*, *derived* or *hypothesis* in the report. This is a personal project, not affiliated with any
company. MusicGen weights are CC-BY-NC 4.0 (non-commercial); MusicCaps captions are CC BY-SA 4.0.
