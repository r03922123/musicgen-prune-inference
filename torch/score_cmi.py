"""
Add CMI-RM judge scores to an existing corpus (second pass over the final WAVs).

CMI-RM's bundled MuQ needs transformers 4.57 while generation uses 5.17, so this runs in its
own environment (see kaggle_corpus/run.py):
  python torch/score_cmi.py --corpus corpus --repo CMI-RewardBench --ckpt .../model.safetensors
Adds scores.cmi_ali / scores.cmi_qual to every <id>.json that lacks them (atomic rewrite).
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scorers import CmiRm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    m = CmiRm(a.repo, a.ckpt)
    for p in sorted(Path(a.corpus).glob("*.json")):
        rec = json.loads(p.read_text())
        if "cmi_qual" in rec["scores"]:
            continue
        wavs = [wavfile.read(p.with_name(f"{rec['id']}_{r}.wav"))[1].astype(np.float32) / 32767
                for r in range(rec["n"])]
        rec["scores"].update({k: v.tolist() for k, v in m(wavs, rec["prompt"]).items()})
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, p)
        print(rec["id"], "cmi_qual", np.round(rec["scores"]["cmi_qual"], 3).tolist(), flush=True)


if __name__ == "__main__":
    main()
