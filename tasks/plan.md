# Implementation Plan: SPEC 0.5, Day 1

## Overview
Day 1 builds the corpus-first offline evaluation for SPEC section 0.5. The steps:
- generate one unpruned best-of-8 corpus (50 prompts, 30 s, musicgen-small fp16) on Kaggle;
- score every candidate with CLAP at 5-second token prefixes and with a judge at 30 s;
- evaluate pruning schedules offline from those score tables with a unit-tested harness;
- produce a compute-vs-regret frontier with bootstrap 95% CIs.

Row-id-keyed sampling makes the offline evaluation exact: a pruned run's survivors equal the
matching unpruned rows. torch/prune.py and the planned pytest suite prove that.

## Architecture Decisions
- **Offline from one corpus.** Every schedule is simulated from the score tables. The GPU
  never runs a pruned generation for evaluation.
- **CLAP is scored at 5, 10, 15, 20, 25 and 30 s prefixes.** SPEC 0.5 says 10 s and 20 s, but
  the frontier needs a 5 s first checkpoint and the final pick needs 30 s. Each prefix is decoded
  from token prefixes with EnCodec, not cut from the final waveform.
- **Schedule family:** first checkpoint c in {5, 10} s, checkpoints every c seconds, keep fraction
  f in {0.5, 0.65, 0.75}, keep = ceil(alive x f). Pruning stops at 1 survivor or 30 s. The winner
  is the 30 s CLAP argmax among the survivors, using the same tie rule as prune.py (stable sort,
  row order kept).
- **Regret baseline:** "best of 8" = the unpruned winner picked by the same pruner (CLAP at
  30 s): regret = judge(baseline winner) - judge(pruned winner). Oracle regret
  (judge max - judge(pruned winner)) is also reported.
- **Metric definitions:**
  - top-2 false-kill: the judge's top-2 rows that did not survive to 30 s;
  - Spearman: between the pruner score at the first checkpoint and the 30 s judge score, per
    prompt over the 8 candidates;
  - compute: candidate-seconds / (8 x 30).
- **Bootstrap:** resample prompts, 10,000 resamples, fixed seed, percentile 95% CI.
- **Judges:** CMI-RM musicality is primary if it runs on a T4 within the 1 h timebox; otherwise
  Audiobox Aesthetics, with the switch recorded. Audiobox (all 4 axes) and CMI-RM alignment are
  logged whenever available, so the fallback costs no extra run.
- **Corpus layout (per prompt):**
  - tokens: int16 [8, K, 1500], committed (small);
  - scores: JSON, committed;
  - WAVs: 16-bit, Kaggle output only, for the Day 2 listening test, never committed.
- **Kaggle single-file limit:** bundle.py is generalised to embed any list of files (sources +
  prompt lists) into a job's run.py.
- **Tests:** they live in torch/tests/, and a conftest puts torch/ first on sys.path, so
  `import prune` gets the torch port, not the MLX file of the same name. The MLX self-check
  stays `python prune.py` (mlx is not installable on Linux CI).

## Task List (order = execution order; one commit per task)

### Phase A: Foundation
- [ ] T1 Commit SPEC + plan, git push master
- [ ] T2 pytest suite from the torch self-checks, incl. noise-keying, KV-cache and cross-attention mutation tests

### Phase B: Corpus path (started early so the GPU runs while evaluate.py is written)
- [ ] T3 prompts: 40 MusicCaps (seeded, licence recorded) + 10 slow-build red-team, committed
- [ ] T4 scorers.py (CLAP chunked, Audiobox, CMI-RM) + generate_corpus.py (resumable), local tiny-duration run
- [ ] T5 Kaggle corpus job + 2-prompt smoke test; CMI-RM timebox (<= 1 h), decision recorded

### Checkpoint B
- [ ] smoke outputs downloaded, score JSON schema confirmed, judge decided

### Phase C: Harness (TDD, written while the GPU runs)
- [ ] T6 evaluate.py part 1: schedule simulation + compute (tests first, hand-computed toy)
- [ ] T7 evaluate.py part 2: regret, P(same winner), false-kill, Spearman, bootstrap CI (tests first)
- [ ] T8 launch the full 50-prompt corpus on Kaggle (background)

### Phase D: Frontier
- [ ] T9 frontier.py: grid over schedules -> results JSON + figure, red-team separate; tested on a synthetic corpus
- [ ] T10 run on the real corpus, commit score tables + results, pick the recommended schedule

### Checkpoint: Day 1 complete (stop here, summarise for Jie)
- [ ] all tests pass; results files committed; every number traceable to a file

## Risks and Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| CMI-RM needs Python 3.10 / pinned deps; Kaggle is 3.13 | High | Hard 1 h timebox, then Audiobox primary; Audiobox scores always computed |
| `pip install audiobox_aesthetics` changes Kaggle's torch | Med | Check torch version after install in the smoke test; `--no-deps` + explicit deps if needed |
| A spare CPU on CI/local for real-model tests | Low | Unit tests use the tiny random model only; real model only on Kaggle |
| 40 prompts give wide CIs | Med | Report n with every claim; "no loss detected on 40 prompts", not "no loss" |
| Kaggle session dies mid-corpus | Low | Per-prompt output files; rerun skips finished prompts |

## Open Questions (proceeding with the defaults above; flag if wrong)
- Primary judge axis: musicality (default) vs alignment vs the mean of both.
- Regret vs the CLAP-selected best-of-8 (default) or vs the judge oracle.
