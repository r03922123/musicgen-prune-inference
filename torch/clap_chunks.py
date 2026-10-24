"""
CLAP cosine of every 10 s chunk of every saved take (the leak-free check, tasks/decisions.md
2026-10-23). CPU is fine; no generation.

  python torch/clap_chunks.py --corpus data/corpus --wavs data/wavs --out results/clap_chunks.json

Writes {id: [[chunk1, chunk2, chunk3] for each of the 8 takes]}, resumable. Also re-derives the
corpus clap[30] from the same embeddings as a pipeline check (max abs diff is reported).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from scorers import Clap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--wavs", default="data/wavs")
    ap.add_argument("--out", default="results/clap_chunks.json")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    clap, worst = Clap(a.device), 0.0
    for p in sorted(Path(a.corpus).glob("*.json")):
        r = json.loads(p.read_text())
        if r["id"] in res:
            continue
        wavs = [wavfile.read(Path(a.wavs) / f"{r['id']}_{i}.wav")[1].astype(np.float32) / 32768
                for i in range(r["n"])]
        with torch.inference_mode():                              # embed once, derive both
            t, e, owner = clap._embed(wavs, r["prompt"])
        own = owner.cpu().numpy()
        cos = (e @ t.T)[:, 0].float().cpu().numpy()
        res[r["id"]] = [cos[own == b].tolist() for b in range(len(wavs))]
        clip = torch.stack([e[owner == b].mean(0) for b in range(len(wavs))])
        c30 = ((clip / clip.norm(dim=-1, keepdim=True)) @ t.T)[:, 0].float().cpu().numpy()
        worst = max(worst, float(np.abs(c30 - np.array(r["scores"]["clap"]["30"])).max()))
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        tmp.replace(out)
        print(f"{r['id']}: done ({len(res)} prompts), max |clap30 - corpus| so far {worst:.4f}", flush=True)


if __name__ == "__main__":
    main()
