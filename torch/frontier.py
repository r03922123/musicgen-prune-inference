"""
Pruning schedules evaluated offline from the unpruned corpus (SPEC 0.5; Q1/Q2 split in
tasks/decisions.md, 2026-10-18).

  python torch/frontier.py --corpus data/corpus --out results

Pruner: CLAP. Schedules: first checkpoint {5, 10} s (then every first s) x keep {0.5, 0.65, 0.75}.
MusicCaps (mc_*) and red-team (rt_*) prompts are reported separately, each with its n.
Q1 pruning fidelity (judge-free): same-winner rate vs the unpruned CLAP pick, compute.
  Recommended = the cheapest schedule with MusicCaps same-winner >= 0.90 (threshold chosen after
  seeing Day 1 results). Regret / false-kill under the judge are kept per schedule.
Q2 selection value, per judge: CLAP-pick gain vs a random candidate, oracle gain, Spearman.
"""
import argparse, json
from pathlib import Path
from evaluate import aggregate, bootstrap_ci, checkpoints, prompt_metrics, selection_metrics

GRID = [(f, k) for f in (5, 10) for k in (0.5, 0.65, 0.75)]
SETS = {"musiccaps": "mc_", "redteam": "rt_"}
JUDGES = ("cmi_qual", "cmi_ali", "mulan", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")
Q2_JUDGES = ("mulan", "cmi_qual", "cmi_ali", "aes_CE")   # pre-registered (tasks/decisions.md)
SAME_WINNER_MIN = 0.90


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
    ok = [s for s in scheds if s["sets"]["musiccaps"]["same_winner"]["mean"] >= SAME_WINNER_MIN]
    rec = min(ok, key=lambda s: s["sets"]["musiccaps"]["compute"]["mean"])
    q2 = [j for j in judges if j in Q2_JUDGES]
    selection = {j: {s: aggregate([selection_metrics(r["scores"]["clap"]["30"], r["scores"][j]) for r in v])
                     for s, v in split.items()} for j in q2}
    for j in q2:                                   # Bonferroni over the judge readings reported
        for s, v in split.items():
            g = [selection_metrics(r["scores"]["clap"]["30"], r["scores"][j])["gain_vs_random"] for r in v]
            selection[j][s]["gain_vs_random_bonferroni"] = dict(bootstrap_ci(g, alpha=0.05 / len(q2)),
                                                               alpha=0.05 / len(q2))
    return {"pruner": "clap", "judge": judge, "secondary_judges": [j for j in judges if j != judge],
            "n": {s: len(v) for s, v in split.items()}, "schedules": scheds,
            "recommended": {"first": rec["first"], "keep": rec["keep"],
                            "rule": f"cheapest MusicCaps compute with same-winner >= {SAME_WINNER_MIN}"},
            "selection": selection}


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"musiccaps": "#2b6cb0", "redteam": "#c05621"}
    fig, (q1, q2) = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    for s, c in colors.items():
        pts = [x for x in res["schedules"] if s in x["sets"] and x["first"] is not None]
        if not pts:
            continue
        xs = [x["sets"][s]["compute"]["mean"] for x in pts]
        ys = [x["sets"][s]["same_winner"]["mean"] for x in pts]
        err = [[y - x["sets"][s]["same_winner"]["lo"] for x, y in zip(pts, ys)],
               [x["sets"][s]["same_winner"]["hi"] - y for x, y in zip(pts, ys)]]
        q1.errorbar(xs, ys, yerr=err, fmt="o", color=c, capsize=3, alpha=0.9, label=f"{s} (n={res['n'][s]})")
        if s == "musiccaps":
            for x, xv, yv in zip(pts, xs, ys):
                left = (x["first"], x["keep"]) in {(10, 0.5), (10, 0.65)}   # neighbours on the right
                q1.annotate(f"{x['first']}s/{x['keep']}", (xv, yv), fontsize=7, color=c,
                            xytext=(-6, -10) if left else (6, -10), textcoords="offset points",
                            ha="right" if left else "left")
    q1.axhline(SAME_WINNER_MIN, color="gray", lw=0.8, ls="--")
    q1.text(0.36, SAME_WINNER_MIN + 0.015, f"rule: same-winner >= {SAME_WINNER_MIN}", fontsize=7, color="gray")
    q1.set_xlabel("compute (candidate-seconds / 8 x 30 s)")
    q1.set_ylabel("P(pruned winner = unpruned CLAP pick)")
    q1.set_title("Q1 pruning fidelity, 95% bootstrap CI", fontsize=10)
    q1.set_ylim(-0.05, 1.08)
    q1.set_xlim(right=0.88)
    q1.legend(fontsize=8, loc="lower right")
    judges = list(res["selection"])
    for k, (s, c) in enumerate(colors.items()):
        if s not in res["n"]:
            continue
        ys = [k * 0.35 + i for i in range(len(judges))]
        g = [res["selection"][j][s]["gain_vs_random"] for j in judges]
        o = [res["selection"][j][s]["oracle_gain"]["mean"] for j in judges]
        q2.errorbar([x["mean"] for x in g], ys, xerr=[[x["mean"] - x["lo"] for x in g], [x["hi"] - x["mean"] for x in g]],
                    fmt="o", color=c, capsize=3, label=f"CLAP pick, {s} (n={res['n'][s]})")
        q2.scatter(o, ys, marker="|", s=120, color=c, label=f"oracle, {s}")
    q2.axvline(0, color="gray", lw=0.8)
    q2.set_yticks([i + 0.175 for i in range(len(judges))], judges, fontsize=8)
    q2.invert_yaxis()
    q2.set_xlabel("judge score gain over a random candidate (judge units)")
    q2.set_title("Q2 selection value: unpruned CLAP best-of-8", fontsize=10)
    q2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
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
              f"same {mc['same_winner']['mean']:.2f}  false-kill {mc['false_kill']['mean']:.2f}  "
              f"survivors {mc['survivors']['mean']:.1f}")
    print("recommended:", res["recommended"], "n:", res["n"], "judge:", judge)
    for j, sets in res["selection"].items():
        print(f"Q2 {j}: " + "  ".join(
            f"{s} gain {m['gain_vs_random']['mean']:+.3f} [{m['gain_vs_random']['lo']:+.3f}, "
            f"{m['gain_vs_random']['hi']:+.3f}] oracle {m['oracle_gain']['mean']:+.3f} "
            f"rho {m['spearman']['mean']:+.2f} [{m['spearman']['lo']:+.2f}, {m['spearman']['hi']:+.2f}]"
            for s, m in sets.items()))


if __name__ == "__main__":
    main()
