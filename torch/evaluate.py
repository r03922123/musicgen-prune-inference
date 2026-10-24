"""
Offline evaluation of pruning schedules from an unpruned corpus.

Row-id-keyed sampling makes a pruned run's survivors identical to the same rows of the
unpruned run (torch/tests/test_prune.py), so any schedule can be simulated exactly from the
corpus score tables, without generating again. Pure functions, numpy only.
"""
import math
import numpy as np


def checkpoints(first, horizon=30, every=None):
    """first, first+every, ... strictly before horizon (the final pick happens at horizon)."""
    return list(range(first, horizon, every or first))


def simulate(early, final, cps, keep_frac, horizon=30):
    """Successive halving on one prompt.
    early: {t: (n,) pruner scores of the t-second prefixes}; final: (n,) pruner scores at horizon.
    At each checkpoint keep ceil(alive * keep_frac) best rows (stable: ties keep lower row ids);
    stop pruning once one row is left. Winner = best final pruner score among the survivors.
    compute = candidate-seconds generated / (n * horizon)."""
    n = len(final)
    alive, log, spent, prev = list(range(n)), {}, 0.0, 0
    for t in cps:
        if len(alive) == 1:
            break
        spent += len(alive) * (t - prev)
        keep = max(1, math.ceil(len(alive) * keep_frac))
        order = sorted(alive, key=lambda r: -early[t][r])        # sorted() is stable
        alive = sorted(order[:keep])
        log[t], prev = alive, t
    spent += len(alive) * (horizon - prev)
    winner = max(alive, key=lambda r: final[r])                  # max() keeps the first on ties
    return {"alive": log, "survivors": alive, "winner": winner, "compute": spent / (n * horizon)}


def fidelity(early, final, cps, keep_frac, horizon=30):
    """One prompt, one schedule: does pruning keep the take unpruned best-of-n picks (by the same
    pruner scores), and at what compute?"""
    r = simulate(early, final, cps, keep_frac, horizon)
    base = simulate(early, final, [], keep_frac, horizon)["winner"]
    return {"compute": r["compute"], "same_winner": float(r["winner"] == base)}


def bootstrap_ci(x, n_boot=10_000, seed=0, alpha=0.05):
    """Mean with a percentile bootstrap CI over prompts. NaNs dropped."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    means = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def aggregate(rows, **kw):
    """[fidelity(...)] -> {metric: bootstrap_ci over prompts}."""
    return {k: bootstrap_ci([r[k] for r in rows], **kw) for k in rows[0]}
