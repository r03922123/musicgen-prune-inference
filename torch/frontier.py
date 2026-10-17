"""
Compute-vs-regret frontier over pruning schedules (SPEC 0.5), from the unpruned corpus.

  python torch/frontier.py --corpus data/corpus --out results

Pruner: CLAP. Schedules: first checkpoint {5, 10} s (then every first s) x keep {0.5, 0.65, 0.75}.
MusicCaps (mc_*) and red-team (rt_*) prompts are reported separately, each with its n.
Recommended = the cheapest schedule whose 95% CI on MusicCaps regret does not lie above 0
("no loss detected on n prompts", not "no loss").
"""
import argparse, json
from pathlib import Path
from evaluate import aggregate, checkpoints, prompt_metrics

GRID = [(f, k) for f in (5, 10) for k in (0.5, 0.65, 0.75)]
SETS = {"musiccaps": "mc_", "redteam": "rt_"}
JUDGES = ("cmi_qual", "cmi_ali", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")


def load_corpus(d):
    return [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))]


def _rows(recs, judge, cps, keep):
    out = []
    for r in recs:
        clap = {int(t): v for t, v in r["scores"]["clap"].items()}
        out.append(prompt_metrics(clap, clap[30], r["scores"][judge], cps, keep))
    return out


def frontier(recs, judge):
    judges = [j for j in JUDGES if all(j in r["scores"] for r in recs)]
    split = {s: [r for r in recs if r["id"].startswith(p)] for s, p in SETS.items()}
    split = {s: v for s, v in split.items() if v}
    scheds = []
    for first, keep in [(None, None)] + GRID:
        cps = checkpoints(first) if first else []
        sets = {s: aggregate(_rows(v, judge, cps, keep or 1.0)) for s, v in split.items()}
        secondary = {j: {s: aggregate(_rows(v, j, cps, keep or 1.0))["regret"] for s, v in split.items()}
                     for j in judges if j != judge}
        scheds.append({"first": first, "keep": keep, "checkpoints": cps, "sets": sets, "secondary": secondary})
    ok = [s for s in scheds if s["sets"]["musiccaps"]["regret"]["lo"] <= 0]
    rec = min(ok, key=lambda s: s["sets"]["musiccaps"]["compute"]["mean"])
    return {"pruner": "clap", "judge": judge, "secondary_judges": [j for j in judges if j != judge],
            "n": {s: len(v) for s, v in split.items()}, "schedules": scheds,
            "recommended": {"first": rec["first"], "keep": rec["keep"],
                            "rule": "cheapest MusicCaps compute with regret CI lo <= 0"}}


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (s, c), dx in zip({"musiccaps": "#2b6cb0", "redteam": "#c05621"}.items(), (0, 0.004)):
        pts = [x for x in res["schedules"] if s in x["sets"]]
        if not pts:
            continue
        xs = [x["sets"][s]["compute"]["mean"] + dx for x in pts]
        ys = [x["sets"][s]["regret"]["mean"] for x in pts]
        err = [[y - x["sets"][s]["regret"]["lo"] for x, y in zip(pts, ys)],
               [x["sets"][s]["regret"]["hi"] - y for x, y in zip(pts, ys)]]
        ax.errorbar(xs, ys, yerr=err, fmt="o", color=c, capsize=3, label=f"{s} (n={res['n'][s]})")
        for x, xv, yv in zip(pts, xs, ys):
            ax.annotate("unpruned" if x["first"] is None else f"{x['first']}s/{x['keep']}",
                        (xv, yv), fontsize=7, xytext=(4, 3), textcoords="offset points", color=c)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("compute (candidate-seconds / 8 x 30 s)")
    ax.set_ylabel(f"regret, {res['judge']} (best-of-8 - pruned)")
    ax.set_title("CLAP-pruned best-of-8: compute vs regret, 95% bootstrap CI", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--judge", default="auto", help="score key; auto = cmi_qual if present, else aes_CE")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    recs = load_corpus(a.corpus)
    judge = a.judge if a.judge != "auto" else (
        "cmi_qual" if all("cmi_qual" in r["scores"] for r in recs) else "aes_CE")
    res = frontier(recs, judge)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "frontier.json").write_text(json.dumps(res, indent=1))
    plot(res, out / "frontier.png")
    for s in res["schedules"]:
        mc = s["sets"]["musiccaps"]
        print(f"first={s['first']} keep={s['keep']}: compute {mc['compute']['mean']:.3f}  "
              f"regret {mc['regret']['mean']:+.3f} [{mc['regret']['lo']:+.3f}, {mc['regret']['hi']:+.3f}]  "
              f"same {mc['same_winner']['mean']:.2f}  false-kill {mc['false_kill']['mean']:.2f}")
    print("recommended:", res["recommended"], "n:", res["n"], "judge:", judge)


if __name__ == "__main__":
    main()
