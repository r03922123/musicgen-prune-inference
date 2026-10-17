"""
Offline evaluation of pruning schedules from an unpruned corpus (SPEC 0.5).

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
