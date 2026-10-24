"""
Pruning schedules evaluated offline from the unpruned corpus (tasks/decisions.md, 2026-10-18).

  python torch/frontier.py --corpus data/corpus --out results

Pruner: CLAP. Schedules: first checkpoint {5, 10} s (then every first s) x keep {0.5, 0.65, 0.75}.
MusicCaps (mc_*) and slow-build (rt_*) prompts are reported separately, each with its n.
Per schedule: compute, and the same-winner rate vs the unpruned CLAP pick, with bootstrap CIs.
Recommended = the cheapest schedule with MusicCaps same-winner >= 0.90 (threshold chosen after
seeing the results).
"""
import argparse, json
from pathlib import Path
from evaluate import aggregate, checkpoints, fidelity

GRID = [(f, k) for f in (5, 10) for k in (0.5, 0.65, 0.75)]
SETS = {"musiccaps": "mc_", "redteam": "rt_"}
SAME_WINNER_MIN = 0.90


def load_corpus(d):
    return [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))]


def frontier(recs):
    split = {s: [r for r in recs if r["id"].startswith(p)] for s, p in SETS.items()}
    split = {s: v for s, v in split.items() if v}
    scheds = []
    for first, keep in [(None, None)] + GRID:
        cps = checkpoints(first) if first else []
        sets = {}
        for s, v in split.items():
            rows = []
            for r in v:
                clap = {int(t): x for t, x in r["scores"]["clap"].items()}
                rows.append(fidelity(clap, clap[30], cps, keep or 1.0))
            sets[s] = aggregate(rows)
        scheds.append({"first": first, "keep": keep, "checkpoints": cps, "sets": sets})
    ok = [s for s in scheds if s["sets"]["musiccaps"]["same_winner"]["mean"] >= SAME_WINNER_MIN]
    rec = min(ok, key=lambda s: s["sets"]["musiccaps"]["compute"]["mean"])
    return {"pruner": "clap", "n": {s: len(v) for s, v in split.items()}, "schedules": scheds,
            "recommended": {"first": rec["first"], "keep": rec["keep"],
                            "rule": f"cheapest MusicCaps compute with same-winner >= {SAME_WINNER_MIN}"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    res = frontier(load_corpus(a.corpus))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "frontier.json").write_text(json.dumps(res, indent=1))
    for s in res["schedules"]:
        print(f"first={s['first']} keep={s['keep']}: " + "  ".join(
            f"{name} compute {m['compute']['mean']:.3f} same {m['same_winner']['mean']:.2f} "
            f"[{m['same_winner']['lo']:.2f}, {m['same_winner']['hi']:.2f}]" for name, m in s["sets"].items()))
    print("recommended:", res["recommended"], "n:", res["n"])


if __name__ == "__main__":
    main()
