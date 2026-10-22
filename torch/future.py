"""
C1 future-only check (pre-registered in tasks/decisions.md, 2026-10-23).

  python torch/future.py --corpus data/corpus --chunks results/clap_chunks.json --out results

Q1 asks whether pruning keeps CLAP best-of-8's 30 s pick; that target shares chunk 1 (0-10 s) with
the 10 s checkpoint score. Here the target is the take CLAP likes best on the last 10 s chunk
(20-30 s), which no testable schedule has heard. Primary: kept - k/8 per schedule, bootstrap CI
over prompts. Secondary: within-prompt Spearman of early CLAP vs later chunks.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from evaluate import aggregate, bootstrap_ci, checkpoints, future_metrics
from frontier import GRID, SETS, load_corpus


def rho(x, y):
    with np.errstate(invalid="ignore"):
        return float(spearmanr(x, y)[0]) if np.ptp(x) and np.ptp(y) else float("nan")


def run(recs, chunks):
    split = {s: [r for r in recs if r["id"].startswith(p)] for s, p in SETS.items()}
    split = {s: v for s, v in split.items() if v}
    out = {"target": "best CLAP cosine on the 20-30 s chunk", "n": {s: len(v) for s, v in split.items()},
           "schedules": [], "predictability": {}}
    for first, keep in GRID:
        cps, sets = checkpoints(first), {}
        for s, v in split.items():
            rows = []
            for r in v:
                clap = {int(t): x for t, x in r["scores"]["clap"].items()}
                rows.append(future_metrics(clap, clap[30], chunks[r["id"]], cps, keep))
            if not all(m["testable"] for m in rows):
                sets[s] = "not testable: a checkpoint after 20 s hears the target chunk"
                continue
            sets[s] = aggregate([{k: m[k] for k in ("kept", "chance", "excess", "cut_first")} for m in rows])
        out["schedules"].append({"first": first, "keep": keep, "checkpoints": cps, "sets": sets})
    for s, v in split.items():
        c = {r["id"]: np.asarray(chunks[r["id"]]) for r in v}
        cl = lambda r, t: np.asarray(r["scores"]["clap"][str(t)])
        out["predictability"][s] = {
            "clap10_vs_chunks23": bootstrap_ci([rho(cl(r, 10), c[r["id"]][:, 1:].mean(1)) for r in v]),
            "clap20_vs_chunk3": bootstrap_ci([rho(cl(r, 20), c[r["id"]][:, 2]) for r in v]),
            # exploratory reference, not pre-registered: the overlap itself
            "clap10_vs_chunk1": bootstrap_ci([rho(cl(r, 10), c[r["id"]][:, 0]) for r in v]),
            "clap30_vs_chunk3": bootstrap_ci([rho(cl(r, 30), c[r["id"]][:, 2]) for r in v]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--chunks", default="results/clap_chunks.json")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    recs = load_corpus(a.corpus)
    res = run(recs, json.loads(Path(a.chunks).read_text()))
    Path(a.out, "future_only.json").write_text(json.dumps(res, indent=1))
    f = lambda m: f"{m['mean']:+.2f} [{m['lo']:+.2f}, {m['hi']:+.2f}]"
    for sc in res["schedules"]:
        for s, m in sc["sets"].items():
            if isinstance(m, str):
                print(f"{sc['first']:>2} s / {sc['keep']:<4} {s:9}: not testable")
            else:
                print(f"{sc['first']:>2} s / {sc['keep']:<4} {s:9} (n={m['kept']['n']}): kept {m['kept']['mean']:.2f} "
                      f"vs chance {m['chance']['mean']:.2f}, excess {f(m['excess'])}, cut at first cp {m['cut_first']['mean']:.2f}")
    for s, p in res["predictability"].items():
        print(s, {k: f(v) for k, v in p.items()})


if __name__ == "__main__":
    main()
