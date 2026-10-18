"""
Exploratory (not pre-registered): how much do the scorers agree, between vs within prompts?

  python torch/agreement.py --corpus data/corpus --out results/judge_agreement.json

between: Spearman over prompts of each scorer's per-prompt mean (do they agree on which prompts
         come out well?). within: per-prompt Spearman over the 8 candidates, bootstrap CI over
         prompts (do they agree on which candidate of a prompt is best? this is what selection needs).
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from evaluate import bootstrap_ci

KEYS = ("clap", "mulan", "cmi_qual", "cmi_ali", "aes_CE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="results/judge_agreement.json")
    a = ap.parse_args()
    recs = [json.loads(p.read_text()) for p in sorted(Path(a.corpus).glob("*.json"))]
    get = lambda r, k: r["scores"]["clap"]["30"] if k == "clap" else r["scores"][k]
    keys = [k for k in KEYS if all(k == "clap" or k in r["scores"] for r in recs)]
    out = {"n_prompts": len(recs), "note": "exploratory, not pre-registered", "pairs": {}}
    for x, y in itertools.combinations(keys, 2):
        between = float(spearmanr([np.mean(get(r, x)) for r in recs], [np.mean(get(r, y)) for r in recs])[0])
        within = bootstrap_ci([spearmanr(get(r, x), get(r, y))[0] for r in recs])
        out["pairs"][f"{x}~{y}"] = {"between": between, "within": within}
        print(f"{x:>8} ~ {y:<8} between {between:+.2f}   within {within['mean']:+.2f} "
              f"[{within['lo']:+.2f}, {within['hi']:+.2f}] (n={within['n']})")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
