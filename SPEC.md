# SPEC: cheaper best-of-N music generation at equal quality

Status: draft 2026-10-17, 2-day scope active (section 0.5). Owner: Jie. Implementer: Claude Code (agent-skills workflow).
Context: rebuild of a pruning prototype from KKCompany (2023-2025). Everything here is
post-employment work and is presented as a personal project, never as KKCompany work.

## 0. Outcome (the definition of done)

Answer, with evidence a skeptical reviewer would accept:

> For MusicGen best-of-8, how much generation compute can early pruning save with no
> measurable loss in quality, as judged by an independent reward model and by blind human
> listening, and which early scorer makes that possible?

Deliverables:
1. A compute-vs-quality Pareto frontier over pruning schedules x early scorers, with 95%
   bootstrap confidence intervals.
2. A recommended scorer + schedule, and the prompts where it breaks (red-team set).
3. A latency section: where per-step time goes (profiled, not guessed) and how much removing
   launch overhead raises the pruning speedup.
4. README, public write-up and a resume Projects bullet where every number traces to a
   committed results file.

Non-goals: production serving at scale, new generator training (except optional DPO),
an audiocraft port, HeartMuLa.

## 0.5 Active scope: 2-day plan (supersedes sections 3 and 5 until done)

Constraint: this project gets 2 working days, because the other past projects also need a
rebuild this week. The design principles stay; sample sizes and breadth shrink. Every claim
is reported with its sample size ("on 50 prompts"), never generalised beyond it.

What is kept: corpus-first offline evaluation, judge independence (CLAP prunes, a different
model judges), red-team prompts, bootstrap CIs, a blind human check, a measured latency
breakdown.

What is cut or deferred: 200 -> 50 prompts (40 MusicCaps + 10 red-team); 3 pruners -> CLAP only
(Audiobox / CMI-RM as pruners deferred); CUDA graphs deferred (profiling kept); CI kept only if
it takes under 20 minutes; all optional items (O1-O4) deferred.

Judge choice, timeboxed: try CMI-RM on the 2-prompt smoke test for at most 1 hour. If it does
not run cleanly on a T4, use Audiobox Aesthetics as the judge and record the switch.

### Day 1
- [ ] (0.5 h) git push; one pytest file from the self-checks, including the mutation tests
- [ ] (1.5 h) torch/evaluate.py: score tables + schedule -> compute, regret, P(same winner),
      top-2 false-kill, Spearman, bootstrap CI. Unit tests on a hand-computed toy case.
- [ ] (0.5 h) prompts: 40 MusicCaps captions (fixed seed, licence checked) + 10 slow-build
      red-team prompts, committed
- [ ] (0.5 h) generate_corpus.py: unpruned N=8, 30 s, save tokens + WAVs, resumable;
      CLAP scores at 10 s and 20 s prefixes; judge score at 30 s
- [ ] Kaggle: 2-prompt smoke test (judge timebox here), then all 50 prompts in the background
      (estimated 1-2 GPU hours). Write evaluate/report code while it runs.
- [ ] (1 h) frontier: first checkpoint {5, 10} s x keep fraction {0.5, 0.65, 0.75}, compute vs
      regret with 95% CIs; red-team reported separately; pick a recommended schedule

### Day 2
- [ ] (1 h) blind listening: 20 prompts where the pruned winner differs from the best-of-8
      winner; randomised, blind; binomial test. State that 20 pairs only detect large effects.
- [ ] (1 h) torch.profiler: 50 decode steps at batch 1 and 8, GPU kernel time vs CPU overhead;
      confirm or refute the 18 ms launch-overhead hypothesis
- [ ] (1.5 h) README: results table, frontier figure, profiler breakdown, limitations
- [ ] (1 h) resume Projects bullet, 2-minute interview narrative (Jie writes, Claude edits),
      update the public article's numbers
- [ ] /review in a fresh session; fix findings; commit

## 1. Principles

- Altitude: every task must move the frontier or its credibility. Anything else is cut.
- Depth: work like an inference/evaluation researcher: >= 200 prompts, bootstrap CIs,
  a profiler instead of hypotheses, one fixed evaluation harness.
- Verification: the scorer that prunes is never the judge that grades it. Tests run in CI.
- Range: pruning is successive halving (as in Hyperband); borrow bootstrap and binomial tests
  from statistics; red-team with slow-build genres.
- Every claim in README is labelled measured, derived, or hypothesis.

## 2. Fixed experimental design

- Generator: facebook/musicgen-small, fp16, 30 s, CFG 3.0, top-k 250, temperature 1.0, N = 8,
  sampling noise keyed by original row id (already implemented in torch/prune.py).
- Prompts: 200 sampled (fixed seed) from MusicCaps captions, list committed; plus 30 red-team
  slow-build prompts (orchestral build-up, post-rock crescendo, ambient intro then drop,
  solo intro then full band). Verify the MusicCaps licence before committing captions.
- Corpus: generate once, unpruned. Save token arrays (int16, [8, K, T] per prompt) and 30 s
  WAVs. Every schedule is evaluated offline from this corpus: because noise is row-id keyed,
  a pruned run's survivors equal the corresponding unpruned rows (proved by the self-check).
- Early scorers (pruners): CLAP (chunked mean, current design), Audiobox Aesthetics,
  CMI-RM (HaiwenXia/CMI-RM). Score every candidate at 5, 10, 15, 20, 25 s by decoding token
  prefixes, not by truncating the final waveform.
- Judges (quality of the final 30 s clip): CMI-RM (primary), Audiobox Aesthetics (secondary),
  blind human A/B (ground truth). A pruner never judges itself: when CMI-RM prunes,
  Audiobox and humans judge.
- Schedules (successive halving family): first checkpoint {5, 10, 15} s,
  keep fraction {0.5, 0.65, 0.75}, checkpoints every 5 or 10 s until 1 survivor or 30 s.
- Metrics per (pruner, schedule): compute in candidate-seconds relative to 8 x 30 s;
  regret = judge(best of 8) - judge(pruned winner); P(same winner); top-2 false-kill rate;
  Spearman(early score, final judge score). Mean with 95% bootstrap CI over prompts.
  Red-team set reported separately.
- Human test: 40 prompts where the recommended pruned winner differs from the full best-of-8
  winner. Blind, randomised order. Binomial test against 50%.

## 3. Full scope phases (deferred; reference design)

### P0 Foundations
- Push master. Convert the self-checks into a pytest suite: row-id noise, KV-cache and
  cross-attention cache slicing (including the mutation tests that must fail), undelay.
- GitHub Actions: run the CPU test suite with a tiny random model on every push.
- torch/evaluate.py: corpus score tables + schedule -> metrics JSON. Pure functions, unit-tested
  against hand-computed toy cases.
- Exit: CI green; harness reproduces known results on a toy corpus.

### P1 Corpus and scores
- torch/generate_corpus.py on Kaggle: resumable, sharded by prompt (12 h sessions,
  about 30 GPU h per week).
- Scorer wrappers for CLAP, Audiobox Aesthetics, CMI-RM. Read CMI-RM's expected sample rate and
  max duration from its repo; chunk if needed. If CMI-RM cannot run on a T4, stop and report.
- Smoke test on 2 prompts, then the full set.
- Exit: score tables for every candidate x prefix x scorer committed (JSON/parquet, not WAVs).

### P2 Pareto frontier
- Run the harness over pruners x schedules. Plot compute vs regret with CIs; red-team separately.
- Pick a recommended configuration and name its failure modes.
- Exit: figure, table and a one-paragraph interpretation in README.

### P3 Human blind test
- Export 40 pairs, a simple blind listening script or page, record choices, binomial test.
- Exit: result with p-value in README.

### P4 Latency
- torch.profiler on 50 decode steps at batch 1 and 8: GPU kernel time vs CPU overhead.
- CUDA graphs: torch.compile(mode="reduce-overhead") with a static KV cache of 1500 positions,
  one graph per batch size (8, 4, 2, 1); fallback to manual torch.cuda.CUDAGraph; if both are
  blocked, report why. Self-check must still pass.
- Exit: profiler breakdown; ms/step eager vs graphed; pruned speedup before and after.

### P5 Write-up
- README results; public article update; resume Projects bullet; 2-minute interview narrative
  (Jie writes, Claude edits).

## 4. Optional, ranked

- O1 Learned early-value predictor: small regressor from early-prefix embeddings to the final
  judge score, used as a fourth pruner on the frontier. Strongest Applied Scientist signal.
- O2 Continuous-batching simulator: replay a request stream with the measured ms/step table;
  report throughput and cost per accepted output. MLE signal.
- O3 Minimal serving: FastAPI endpoint for pruned generation, Dockerfile, one run on an AWS g4dn
  spot instance with a budget alarm. Closes the AWS/deployment gap.
- O4 DPO on musicgen-small with CMI-RM-labelled pairs; measure how much N can shrink.
  Needs a rented 24 GB GPU.
- Dropped: audiocraft port, HeartMuLa.

## 5. Full-scope TODO (deferred)

- [ ] P0 git push
- [ ] P0 pytest suite from self-checks (incl. mutation tests)
- [ ] P0 GitHub Actions CPU CI
- [ ] P0 torch/evaluate.py + unit tests
- [ ] P1 prompt set: 200 MusicCaps + 30 red-team, committed
- [ ] P1 generate_corpus.py (resumable, sharded) + Kaggle job
- [ ] P1 scorer wrappers: CLAP, Audiobox Aesthetics, CMI-RM
- [ ] P1 smoke test 2 prompts
- [ ] P1 full corpus + score tables
- [ ] P2 frontier over pruners x schedules with bootstrap CIs
- [ ] P2 red-team analysis, recommended config
- [ ] P3 40-pair blind listening test + binomial test
- [ ] P4 profiler breakdown
- [ ] P4 CUDA graphs + re-benchmark
- [ ] P5 README, article, resume bullet, interview narrative
- [ ] O1 / O2 / O3 / O4 as time allows

## 6. Workflow with agent-skills

- /spec: review this file, do not regenerate it. /plan: break one phase at a time into tasks.
- /build (incremental, one task per commit) with test-driven-development for evaluate.py,
  scorer wrappers and metrics: they are pure functions and easy to test first.
- /test after P0 and whenever a module lands. /review at the end of every phase, ideally in a
  fresh session so the reviewer did not write the code.
- ci-cd-and-automation for GitHub Actions; documentation-and-adrs for one ADR per key decision
  (judge independence, schedule family, corpus-first design).
- Do not use /build auto across phases: Jie approves at each phase exit.
- Not needed: frontend, browser testing, web performance, security hardening.

## 7. Risks

- CMI-RM may not run on a T4 or may expect short clips: chunk, or fall back to Audiobox as the
  primary judge and say so.
- Kaggle quota: shard and resume; corpus generation is the only heavy GPU step.
- Single human rater: blind randomisation reduces bias; report it as a limitation.
- CMI-RM data is CC BY-NC-SA: fine for a non-commercial personal project; note it in README.
