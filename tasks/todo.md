# Day 1 tasks (SPEC 0.5)

## [x] Task 1: Commit SPEC + plan, push master
**Acceptance:** SPEC.md and tasks/ committed; `git status` clean; origin/master == master.
**Verify:** `git log origin/master -1` matches the local HEAD.
**Deps:** none. **Files:** SPEC.md, tasks/*. **Scope:** XS

## [x] Task 2: pytest suite from the self-checks
**Acceptance:**
- [ ] tests: undelay inverts delay; candidates differ; survivors == unpruned rows (token match 1.0); offline rule == online survivors
- [ ] mutation tests: position-keyed noise, wrong KV-cache rows, wrong cross-attention-cache rows each drop the match below 1.0
- [ ] runs on CPU in < 60 s
**Verify:** `uv run pytest torch/tests -q` green; temporarily break `g[ids]` and see the test fail.
**Deps:** T1. **Files:** torch/tests/{conftest,test_prune}.py, pyproject.toml. **Scope:** S

## [x] Task 3: prompt set
**Acceptance:**
- [ ] data/prompts/musiccaps_40.txt sampled with seed 0 from musiccaps-public.csv, with ytid kept for attribution
- [ ] data/prompts/redteam_10.txt: slow-build prompts (orchestral build-up, post-rock crescendo, ambient intro then drop, solo intro then full band)
- [ ] licence (CC BY-SA 4.0) + source recorded in data/prompts/README.md; script reproduces the file byte-for-byte
**Verify:** rerun make_prompts.py -> `git diff --exit-code data/prompts`.
**Deps:** none. **Files:** torch/make_prompts.py, data/prompts/*. **Scope:** S

## [x] Task 4: scorers + generate_corpus.py
**Acceptance:**
- [ ] per prompt: tokens [8,K,1500] int16, 8 WAVs, scores JSON {clap:{5..30}, audiobox:{CE,CU,PC,PQ}, cmi:{mus,ali} if available}
- [ ] CLAP prefixes decoded from token prefixes; resumable (skips prompts with a complete scores file)
- [ ] runs locally end to end at --seconds 2 on 1 prompt (judge scorers may be skipped locally)
**Verify:** local run writes the expected files; rerun skips them.
**Deps:** T3. **Files:** torch/scorers.py, torch/generate_corpus.py. **Scope:** M

## [x] Task 5: Kaggle corpus job, 2-prompt smoke test, CMI-RM timebox
**Acceptance:**
- [ ] kaggle_corpus/ (metadata + run.py bundled) runs 2 prompts on a T4, outputs downloaded
- [ ] CMI-RM either produces scores on the smoke WAVs within 1 h of starting, or the switch to Audiobox is recorded in tasks/decisions.md
**Verify:** downloaded scores JSON has all keys; the timebox start/end times are logged.
**Deps:** T4. **Files:** kaggle_corpus/*, kaggle_job/bundle.py, tasks/decisions.md. **Scope:** M

## Checkpoint B
- [ ] smoke scores schema confirmed; judge decided; torch version unchanged by the installs

## [x] Task 6: evaluate.py part 1, schedule simulation + compute (TDD)
**Acceptance:** tests written first and seen failing; hand-computed toy case (4 rows, cps 10/20, keep 0.5 -> survivors {1,2}->{2}, compute 70/120); keep=ceil rule; ties stable.
**Verify:** `uv run pytest torch/tests/test_evaluate.py -q`.
**Deps:** T2. **Files:** torch/evaluate.py, torch/tests/test_evaluate.py. **Scope:** S

## [x] Task 7: evaluate.py part 2, metrics + bootstrap (TDD)
**Acceptance:** toy case: regret -2, oracle regret 1, same_winner False, false-kill 0.5, Spearman 0.2; bootstrap CI deterministic with a seed and covers the mean; constant input -> zero-width CI.
**Verify:** `uv run pytest torch/tests -q`.
**Deps:** T6. **Files:** torch/evaluate.py, torch/tests/test_evaluate.py. **Scope:** S

## [x] Task 8: full 50-prompt corpus on Kaggle (background)
**Acceptance:** 50 complete scores files downloaded (tokens + scores committed under data/corpus/).
**Verify:** count == 50; every JSON has the full key set.
**Deps:** Checkpoint B. **Scope:** XS (config) + wait

## [x] Task 9: frontier.py
**Acceptance:** grid {5,10} s x {0.5,0.65,0.75} -> results/frontier.json (mean + CI per metric, n per set) + results/frontier.png; MusicCaps and red-team separate; test on a synthetic 3-prompt corpus.
**Verify:** `uv run pytest torch/tests -q`; figure renders.
**Deps:** T7. **Files:** torch/frontier.py, torch/tests/test_frontier.py. **Scope:** S

## [x] Task 10: frontier on the real corpus + recommendation
**Acceptance:** results committed; recommended schedule = lowest compute whose regret CI includes 0 on the 40 MusicCaps prompts; red-team numbers reported next to it.
**Verify:** numbers in the summary match results/frontier.json.
**Deps:** T8, T9. **Scope:** XS

## Checkpoint: end of Day 1, stop and summarise

# Day 2 (tasks/decisions.md, 2026-10-18)
- [x] Decisions 1-4 recorded before new computation
- [x] MuQ-MuLan scoring (CPU + T4 cross-check) and Q2
- [x] CMI-RM scoring in its own venv (v1 setup failure, v2 complete)
- [x] Blind listening script (torch/listen.py); listening itself: pending (Jie)
- [ ] torch.profiler on T4: job pushed, queued on Kaggle; README reports it as pending
- [x] README restructured around Q1/Q2 and SPEC 0.6
