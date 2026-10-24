"""
Score the saved final WAVs of a corpus with one judge (second pass, no regeneration).

  python torch/score_judge.py score --judge mulan --corpus data/corpus --wavs wavs --out mulan.json
  python torch/score_judge.py score --judge cmi --repo CMI-RewardBench --ckpt .../model.safetensors ...
  python torch/score_judge.py merge results/judge_mulan.json --corpus data/corpus

Both judges build MuQ's Wav2Vec2Conformer, which transformers 5.x rejects, so scoring runs in an
environment with transformers 4.57 (see kaggle_judge/run.py). `score` writes {id: {key: [n]}}
(resumable: the file is rewritten after every prompt); `merge` copies those keys into
<corpus>/<id>.json["scores"] so frontier.py reads every judge the same way.
"""
import argparse, json, os
from pathlib import Path
import numpy as np


def atomic_write(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def score(a):
    from scipy.io import wavfile
    from scorers import CmiRm, MuqMulan
    m = MuqMulan(a.device) if a.judge == "mulan" else CmiRm(a.repo, a.ckpt, a.device)
    out = Path(a.out)
    done = json.loads(out.read_text()) if out.exists() else {}
    for p in sorted(Path(a.corpus).glob("*.json")):
        rec = json.loads(p.read_text())
        if rec["id"] in done:
            continue
        wavs = [wavfile.read(Path(a.wavs) / f"{rec['id']}_{r}.wav")[1].astype(np.float32) / 32767
                for r in range(rec["n"])]
        done[rec["id"]] = {k: v.tolist() for k, v in m(wavs, rec["prompt"]).items()}
        atomic_write(out, done)
        print(rec["id"], {k: np.round(v, 3).tolist() for k, v in done[rec["id"]].items()}, flush=True)


def merge(a):
    scores = json.loads(Path(a.scores).read_text())
    for pid, s in scores.items():
        p = Path(a.corpus) / f"{pid}.json"
        rec = json.loads(p.read_text())
        rec["scores"].update(s)
        atomic_write(p, rec)
    print(f"merged {sorted({k for s in scores.values() for k in s})} into {len(scores)} prompts")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--judge", choices=("mulan", "cmi"), required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--wavs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--repo", default="CMI-RewardBench")
    s.add_argument("--ckpt", default="CMI-RewardBench/baselines/model/model.safetensors")
    m = sub.add_parser("merge")
    m.add_argument("scores")
    m.add_argument("--corpus", required=True)
    a = ap.parse_args()
    score(a) if a.cmd == "score" else merge(a)


if __name__ == "__main__":
    main()
