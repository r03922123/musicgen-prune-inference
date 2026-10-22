# Decisions log (SPEC 0.5)

## 2026-10-17

**CLAP prefixes at 5/10/15/20/25/30 s, not only 10/20 s.** The frontier needs a 5 s first
checkpoint and the final pick needs 30 s; extra prefixes are cheap (EnCodec decode + CLAP).

**Regret baseline = the unpruned best-of-8 picked by the same pruner (CLAP at 30 s).** This is
what "no loss from pruning" means. Oracle regret (judge max - pruned winner) is reported too.

**Prompts keep MusicCaps' "low quality recording" captions (11 of 40).** They are part of the
caption distribution; this is why Audiobox PQ (production quality) is not used as a judge axis.
Fallback judge axis = Audiobox CE (content enjoyment).

**Test model init std 0.2 (torch/prune.py random_model).** At HF's 0.02 a wrong self-attention
KV cache moves logits by ~1e-3 and never changes a token, so the cache mutation test could not
fail. Found by the mutation test itself.

**CMI-RM runs in its own venv, as a second pass over the WAVs.** Smoke v1 (19:18): CMI-RM's
bundled MuQ builds a Wav2Vec2Conformer from an EasyDict config, which transformers 5.17's
masking code rejects (`'EasyDict' object has no attribute '_attn_implementation'`); CMI-RM pins
transformers 4.57.3. Generation keeps 5.17 (the version the self-check passed on); CMI-RM gets a
`--system-site-packages` venv with transformers 4.57.3 (shares Kaggle's torch 2.11).

**CMI-RM timebox: 19:09-20:09. Outcome: switched the judge to Audiobox Aesthetics CE.**
CMI-RM did not run cleanly on a T4 within the hour: v1 failed (above); v2 with the venv fix
sat QUEUED on Kaggle from 19:30 past 20:09 and never ran. Per SPEC 0.5 the primary judge is
now Audiobox CE (content enjoyment); CU/PC/PQ are logged as secondary. CMI-RM is removed from
the full run (USE_CMI = False), so a second judge cannot be picked after seeing results. The
venv route (torch/score_cmi.py) is kept for a later attempt outside the 2-day scope.
Judge independence still holds: CLAP prunes, Audiobox (a different model, prompt-free) judges.

## 2026-10-18: Day 1 results, read before using results/frontier.json

**Full corpus: 50/50 prompts, 45.9 min on a T4, no errors** (40 MusicCaps + 10 red-team).

**The recommendation rule is too weak, and I did not change it after seeing results.**
"Cheapest schedule whose regret CI lo <= 0" rewards wide CIs: it picks 5 s / keep 0.5
(compute 0.354) with MusicCaps regret +0.042 [-0.082, +0.172], P(same winner) 0.62, and on the
red-team set regret +0.256 [-0.156, +0.778] with P(same winner) 0.30. A sounder rule bounds
the CI's upper end by a pre-declared margin. Jie to choose the margin before Day 2.

**The judge barely sees what CLAP selects for (measured, n = 40 MusicCaps):**
- CLAP's unpruned best-of-8 pick vs a random candidate, Audiobox CE: +0.19 [-0.04, +0.42]
  (red-team, n = 10: +0.34 [+0.05, +0.61]); oracle vs random: +1.12.
- Spearman(CLAP at first checkpoint, CE) about 0.08 [-0.04, 0.21].
- CLAP agrees with itself: Spearman(CLAP 10 s, CLAP 30 s) = 0.69 [0.63, 0.75].
So "no measurable regret" mostly says pruning preserves CLAP's choice, not that it preserves
quality: selection itself gives no measurable CE gain on these 40 prompts. This weakens the
Day 2 listening test as planned (it would compare two picks the judge can't tell apart).

## 2026-10-18: Day 2 decisions, recorded before any new computation

Recorded before the MuQ-MuLan / CMI-RM scoring, the listening test and the profiler, so none of
them can shape these choices. Inputs seen so far: results/frontier.json (Day 1) only.

**1. The analysis is split into two questions.**
- Q1 pruning fidelity: does the pruned schedule keep the unpruned CLAP best-of-8 pick, and at
  what compute? Metrics: same-winner rate vs the unpruned CLAP pick, compute. Judge-free.
- Q2 selection value: does the CLAP pick beat a random candidate under an independent judge?
  Metrics: CLAP-pick gain vs random (mean judge score of the pick minus the mean over all 8),
  oracle gain (judge max minus the same mean), Spearman(CLAP 30 s, judge), bootstrap CIs.
Regret under a judge (Day 1's headline) is still reported but is no longer the decision metric:
it mixes Q1 and Q2, and Day 1 showed Q2 is near zero under Audiobox CE.

**2. Q1 recommendation rule: the cheapest schedule with MusicCaps same-winner >= 0.90.**
This threshold was set AFTER seeing the Day 1 results (frontier.json), so it is not
pre-registered; it replaces the Day 1 rule (regret CI lo <= 0), which rewarded wide CIs.
The red-team set (n = 10) is reported separately and does not enter the rule.

**3. False-kill caveat.** Top-2 false-kill counts prompts where at least one of the judge's top-2
candidates is pruned. A schedule that ends with 1 survivor must kill at least one of the two,
so its false-kill is >= 0.5 by construction. False-kill is only compared between schedules
with the same final survivor count.

**4. Q2 judges, pre-registered:**
- MuQ-MuLan text-audio similarity (Jan 2025): the then-era alignment judge (SPEC 0.6 (a)).
- CMI-RM (Mar 2026): the now-era judge (SPEC 0.6 (b)), as a background scoring pass in its
  own venv; if it has not finished when the README is written it is reported as pending.
- Audiobox Aesthetics CE stays as the enjoyment axis.
No judge is dropped or added after seeing its numbers.

## 2026-10-18: Q2 under MuQ-MuLan (step 5)

**Where the scores came from.** The Kaggle MuQ-MuLan job queued for about 25 min, so the same
script (torch/score_judge.py, transformers 4.57.3, muq 0.1.0) scored all 400 WAVs on the Mac's
CPU: results/judge_mulan.json. The Kaggle T4 run is a cross-check of the same numbers.

**Result (n = 40 MusicCaps / 10 red-team): CLAP's pick does not beat a random candidate under
MuQ-MuLan.** Gain -0.008 [-0.030, +0.015] (oracle +0.111); red-team +0.009 [-0.055, +0.073].
Within-prompt Spearman(CLAP 30 s, MuQ-MuLan) -0.05 [-0.18, +0.08] on MusicCaps.
Exploratory check (results/judge_agreement.json, not pre-registered): the two text-audio models
agree mildly on which PROMPTS come out well (between-prompt rho +0.35, n = 50) but not on which
CANDIDATE of a prompt is best (within-prompt rho +0.01 [-0.11, +0.13]). The within-prompt
choice is the decision that selection needs.
Cross-check: the Kaggle T4 run (results/judge_mulan_t4.json, 5.0 min of scoring, torch 2.11 +
transformers 4.57.3) matches the CPU scores to max |diff| 2.5e-5, same argmax on 50/50 prompts.

## 2026-10-18: Q2 under CMI-RM (step 6)

CMI-RM v1 failed in setup (Kaggle's Python has no ensurepip, so `venv` failed); v2 creates
the venv `--without-pip` and installs into it with the main pip's `--python`. 50/50 prompts
in 5.5 min on a T4 (results/judge_cmi.json, judge_cmi_summary.json).

Result, MusicCaps n = 40: CLAP-pick gain vs random under CMI-RM quality +0.157 [+0.010, +0.300]
(oracle +0.590); alignment +0.110 [-0.004, +0.219]. This is the only MusicCaps reading of
the four pre-registered ones whose 95 % CI excludes 0, and it does not survive Bonferroni over the
four (98.75 % CI [-0.029, +0.340]); reported as a small, marginal gain, not as a finding.
Q1 regret columns keep Audiobox CE as the judge (Day 1 choice); frontier.py is run with
`--judge aes_CE` because `auto` would now pick cmi_qual.

## 2026-10-18: profiler (step 8)

Kaggle queued the job 18:01-19:41; it ran 19:41-19:44 after the self-check passed (token match
1.0). Batch 1: wall 18.7 ms/step, GPU busy 6.5 ms (35 %), ~498 kernels/step at 13 us mean;
batch 8: 22.2 / 8.6 ms (39 %). The launch-overhead hypothesis is confirmed; CUDA graphs stay
deferred (not implemented, per instruction). results/profile_decode_small_float16_pretrained_t4.json.

## 2026-10-20: Eight Takes listening game (Jie's call: a public game, deployed tonight)

- Format: single-elimination bracket over a prompt's 8 takes; rounds on 10 s, 20 s, 30 s clips
  (8 -> 4 -> 2 -> 1) = the 10 s / keep-0.5 schedule with a human pruner. Each round restarts
  from 0 s (Jie's choice: fairer comparison; costs 220 s of listening vs 140 s continuing).
- CLAP is the opponent ("AI picker"), never a referee; the four Q2 judges are the critics in
  the reveal. Same rule as the study: the pruner is never the judge.
- Both takes must be heard for >= 3 s before a choice; "can't decide" = coin flip, logged as a tie.
- Audio: one linear gain per take to -16 LUFS, peak <= -1 dBFS, boost capped at +15 dB
  (one near-silent take, mc_eStzDzEopDI row 0, would otherwise get +27.7 dB of noise).
- Data: game logs stay in the player's browser (localStorage). No analytics, no backend.
- Research add-on: results/bracket.json (all 315 brackets per prompt, CLAP as player): champion
  = unpruned CLAP pick 0.81 [0.72, 0.88] (MusicCaps n = 40), top-k at the same checkpoints 0.875.
- Repo made public (Jie's choice) so GitHub Pages can serve the site; history scanned for
  secrets first (none found).

## 2026-10-23: C1, future-only check of early CLAP (recorded BEFORE computing)

Red-team finding C1: the 30 s CLAP score is the cosine of the re-normalised mean of the three
10 s chunk embeddings (torch/scorers.py:36-47), and the 10 s prefix score is essentially chunk 1
(one prompt: |clap[10] - chunk-1 cosine| <= 0.017). So Q1 same-winner shares chunk 1 between the
early score and the target. Q1 stays valid as *operational fidelity to CLAP best-of-8* and is
relabelled as such; it is not evidence that early audio predicts later audio. This check measures
that, against a target no checkpoint has heard.

- Data: CLAP cosine of each final WAV's three 10 s chunks (0-10, 10-20, 20-30 s), same model and
  preprocessing as the corpus (48 kHz, L2-normalised chunk and text embeddings), all 400 takes ->
  results/clap_chunks.json. Caveat: chunks are cut from the final decode, not decoded prefixes.
- Testable schedules: those whose last pruning checkpoint is <= 20 s (decided by simulate(), not
  hard-coded), so chunk 3 (20-30 s) is unheard. Others are reported as "not testable".
- Target F = the take with the best chunk-3 cosine. The final pick at 30 s hears chunk 3, so it is
  excluded; only the surviving set is scored.
- Primary metric, per testable schedule: kept = 1 if F survives pruning; compared with chance k/8
  (k = final survivors), as the paired difference kept - k/8 with a 95 % bootstrap CI over prompts
  (10,000 resamples, seed 0). MusicCaps (n = 40) and red-team (n = 10) separately.
- Hypothesis: early CLAP carries information about later audio -> kept - k/8 > 0.
- Predictability (secondary): within-prompt Spearman of clap[10] vs mean chunk-2/3 cosine and of
  clap[20] vs chunk-3 cosine, bootstrap CI over prompts.
- Exploratory: share of prompts where F is cut at the first (10 s) checkpoint.
- This is early CLAP vs late CLAP (fidelity of the pruner to itself over time), not quality; the
  pruner-is-never-the-judge rule still applies to quality claims.

Outcome (computed after the entry above was committed in 9cae3fa; results/future_only.json):
hypothesis supported on MusicCaps (n = 40), every testable schedule keeps the 20-30 s favourite
above chance, e.g. 10 s / 0.65: 0.88 vs 0.50, excess +0.38 [+0.28, +0.47]. Red-team (n = 10):
same direction, all CIs reach 0 or below. 5 s / 0.65 and 5 s / 0.75 not testable as pre-specified.
Exploratory reference: clap10 vs chunk-1 within-prompt Spearman +0.99, confirming the overlap.
