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


def prompt_metrics(early, final, judge, cps, keep_frac, horizon=30):
    """One prompt, one schedule. "Best of N" is the unpruned run's pick by the same pruner
    (final pruner argmax), so regret answers "what did pruning cost vs not pruning?".
      regret         judge(unpruned pick) - judge(pruned winner)   (< 0: pruning got lucky)
      oracle_regret  max judge - judge(pruned winner)
      false_kill     share of the judge's top-2 rows that did not survive to the end
      spearman       pruner score at the first checkpoint vs final judge score, over rows
      survivors      rows alive at the end (with 1 survivor, false_kill >= 0.5 by construction)"""
    from scipy.stats import spearmanr
    judge = np.asarray(judge, dtype=float)
    r = simulate(early, final, cps, keep_frac, horizon)
    base = simulate(early, final, [], keep_frac, horizon)["winner"]
    top2 = sorted(range(len(judge)), key=lambda i: -judge[i])[:2]
    rho = float(spearmanr(early[cps[0]], judge)[0]) if cps else float("nan")
    return {"compute": r["compute"],
            "regret": float(judge[base] - judge[r["winner"]]),
            "oracle_regret": float(judge.max() - judge[r["winner"]]),
            "same_winner": float(r["winner"] == base),
            "false_kill": sum(i not in r["survivors"] for i in top2) / 2,
            "spearman": rho,
            "survivors": len(r["survivors"])}


def selection_metrics(final, judge):
    """Q2, one prompt: is the pruner's unpruned best-of-n pick better than a random candidate?
      gain_vs_random  judge(pruner pick) - mean judge (the expectation of a uniform random pick)
      oracle_gain     max judge - mean judge (the most any selector could gain)
      spearman        final pruner score vs judge score, over rows (NaN if either is constant)"""
    from scipy.stats import spearmanr
    judge = np.asarray(judge, dtype=float)
    pick = int(np.argmax(final))                                 # argmax keeps the first on ties
    with np.errstate(invalid="ignore"):
        rho = float(spearmanr(final, judge)[0]) if np.ptp(final) and np.ptp(judge) else float("nan")
    return {"gain_vs_random": float(judge[pick] - judge.mean()),
            "oracle_gain": float(judge.max() - judge.mean()),
            "spearman": rho}


def all_brackets(rows):
    """Every distinct single-elimination bracket over rows (an int n means range(n), n a power
    of 2), as seedings: adjacent pairs meet in round 1, adjacent winners in round 2, ...
    n!/2^(n-1) of them: 1, 3, 315 for n = 2, 4, 8."""
    rows = list(range(rows)) if isinstance(rows, int) else list(rows)
    if len(rows) == 1:
        return [tuple(rows)]
    from itertools import combinations
    first, rest, out = rows[0], rows[1:], []
    for others in combinations(rest, len(rows) // 2 - 1):     # the half that holds rows[0]
        a = [first, *others]
        b = [r for r in rest if r not in others]
        out += [x + y for x in all_brackets(a) for y in all_brackets(b)]
    return out


def bracket_champion(seeding, rounds):
    """rounds[i]: (n,) scores used in round i (one round per halving); the higher score wins
    each match, ties go to the lower row id."""
    alive = list(seeding)
    for s in rounds:
        alive = [a if (s[a], -a) > (s[b], -b) else b for a, b in zip(alive[::2], alive[1::2])]
    return alive[0]


def bracket_metrics(early, final, judge, cps, horizon=30):
    """The game's pruning (pairwise single elimination, one checkpoint per round, the final at
    horizon) over all distinct brackets, vs top-k successive halving (keep 0.5) at the same
    checkpoints and the unpruned pick, all by the same pruner scores.
      same_as_unpruned / same_as_topk  share of brackets whose champion is that winner
      regret / topk_regret             judge(unpruned pick) - judge(champion), bracket mean / top-k"""
    judge = np.asarray(judge, dtype=float)
    rounds = [early[t] for t in cps] + [final]
    base = simulate(early, final, [], 1.0, horizon)["winner"]
    topk = simulate(early, final, cps, 0.5, horizon)["winner"]
    champs = [bracket_champion(b, rounds) for b in all_brackets(len(final))]
    return {"brackets": len(champs),
            "same_as_unpruned": float(np.mean([c == base for c in champs])),
            "same_as_topk": float(np.mean([c == topk for c in champs])),
            "regret": float(np.mean([judge[base] - judge[c] for c in champs])),
            "topk_regret": float(judge[base] - judge[topk])}


def bootstrap_ci(x, n_boot=10_000, seed=0, alpha=0.05):
    """Mean with a percentile bootstrap CI over prompts. NaNs (e.g. undefined Spearman) dropped."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    means = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def aggregate(rows, **kw):
    """[prompt_metrics(...)] -> {metric: bootstrap_ci over prompts}."""
    return {k: bootstrap_ci([r[k] for r in rows], **kw) for k in rows[0]}
