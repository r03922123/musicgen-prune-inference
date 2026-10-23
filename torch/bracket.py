"""
Knockout-bracket pruning (as a human pairwise study would run it), simulated with CLAP as the player (2026-10-20).

  python torch/bracket.py --corpus data/corpus --out results

Game: 8 takes, single elimination, rounds judged on 10 s, 20 s, then 30 s clips (8 -> 4 -> 2 -> 1),
the same checkpoints as the 10 s / keep 0.5 schedule in frontier.json. A bracket pairs takes at
random, so unlike top-k it can knock out the 2nd-best take in round 1. Over all 315 brackets per
prompt: how often the champion equals the unpruned CLAP pick and the top-k winner, and the judge
regret of each. MusicCaps and red-team reported separately with n.
"""
import argparse, json
from pathlib import Path
from evaluate import aggregate, bracket_metrics
from frontier import Q2_JUDGES, SETS, load_corpus

CPS = [10, 20]


def run(recs):
    out = {"pruner": "clap", "checkpoints": CPS, "rounds_seconds": CPS + [30], "judges": {}}
    split = {s: [r for r in recs if r["id"].startswith(p)] for s, p in SETS.items()}
    split = {s: v for s, v in split.items() if v}
    out["n"] = {s: len(v) for s, v in split.items()}
    for j in Q2_JUDGES:
        if not all(j in r["scores"] for r in recs):
            continue
        out["judges"][j] = {}
        for s, v in split.items():
            rows = []
            for r in v:
                clap = {int(t): x for t, x in r["scores"]["clap"].items()}
                rows.append(bracket_metrics(clap, clap[30], r["scores"][j], CPS))
            out["judges"][j][s] = aggregate(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    res = run(load_corpus(a.corpus))
    Path(a.out, "bracket.json").write_text(json.dumps(res, indent=1))
    for s, n in res["n"].items():
        m = res["judges"]["cmi_qual"][s]
        print(f"{s} (n={n}): champion = unpruned pick {m['same_as_unpruned']['mean']:.2f} "
              f"[{m['same_as_unpruned']['lo']:.2f}, {m['same_as_unpruned']['hi']:.2f}], "
              f"= top-k winner {m['same_as_topk']['mean']:.2f}")
    for j, d in res["judges"].items():
        m = d["musiccaps"]
        print(f"  {j}: bracket regret {m['regret']['mean']:+.3f} [{m['regret']['lo']:+.3f}, "
              f"{m['regret']['hi']:+.3f}], top-k regret {m['topk_regret']['mean']:+.3f}")


if __name__ == "__main__":
    main()
