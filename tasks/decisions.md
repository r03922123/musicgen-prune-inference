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
