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
