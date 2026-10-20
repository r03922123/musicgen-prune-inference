"""
Build the listening game's data and audio (site/, 2026-10-20). Mac, no GPU, no model calls.

  python torch/build_site.py                 # site/data.json + site/audio/<id>_<row>.mp3
  python torch/build_site.py --no-audio      # data.json only

data.json holds, per prompt, the caption, CLAP prefix scores at the game's rounds (10, 20, 30 s)
and the four Q2 judges' scores, plus headline numbers copied from committed results files (each
with its source file and n). Audio: every take loudness-normalised to -16 LUFS with one linear
gain (no compression, so slow builds keep their dynamics), peak-limited, boost capped at
+15 dB, 64 kbps mono MP3.
"""
import argparse, json, re, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = (10, 20, 30)
JUDGES = ("aes_CE", "mulan", "cmi_qual", "cmi_ali")
TARGET_LUFS, PEAK_DBFS, MAX_BOOST_DB = -16.0, -1.0, 15.0


def title(caption, n=64):
    """Short display title: the first clause or sentence, cut at a word boundary."""
    t = re.split(r"(?<=[.;])\s|,\s", caption.strip(), maxsplit=1)[0].rstrip(".;")
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def prompts(recs):
    out = []
    for r in recs:
        s = r["scores"]
        out.append({"id": r["id"], "set": "redteam" if r["id"].startswith("rt_") else "musiccaps",
                    "title": title(r["prompt"]), "caption": r["prompt"], "n": r["n"],
                    "clap": {str(t): s["clap"][str(t)] for t in ROUNDS},
                    "judges": {j: s[j] for j in JUDGES}})
    return out


def headline(results):
    """Numbers shown on the page, copied (not recomputed) from committed results files."""
    R = Path(results)
    fr = json.loads((R / "frontier.json").read_text())
    br = json.loads((R / "bracket.json").read_text())
    pr = json.loads((R / "profile_decode_small_float16_pretrained_t4.json").read_text())
    ag = json.loads((R / "judge_agreement.json").read_text())
    sched = {f"{s['first']}_{s['keep']}": s for s in fr["schedules"] if s["first"]}
    pick = lambda d: {k: d[k] for k in ("mean", "lo", "hi", "n")}
    out = {"schedule_10_0.5": {"source": "results/frontier.json"},
           "selection": {"source": "results/frontier.json"},
           "bracket": {"source": "results/bracket.json"},
           "profile_b1": {"source": "results/profile_decode_small_float16_pretrained_t4.json"},
           "agreement": {"source": "results/judge_agreement.json", "n_prompts": ag["n_prompts"]}}
    for pair in ("clap~mulan", "clap~cmi_qual", "clap~aes_CE"):
        a = ag["pairs"][pair]
        out["agreement"][pair] = {"between": a["between"], "within": pick(a["within"])}
    for st in ("musiccaps", "redteam"):
        m = sched["10_0.5"]["sets"][st]
        out["schedule_10_0.5"][st] = {"compute": pick(m["compute"]), "same_winner": pick(m["same_winner"])}
        b = br["judges"]["cmi_qual"][st]
        out["bracket"][st] = {"same_as_unpruned": pick(b["same_as_unpruned"]),
                              "same_as_topk": pick(b["same_as_topk"])}
    for j in JUDGES:
        sel = fr["selection"][j]["musiccaps"]
        out["selection"][j] = {"gain_vs_random": pick(sel["gain_vs_random"]),
                               "oracle_gain": pick(sel["oracle_gain"])}
    b1 = pr["batches"]["1"]
    out["profile_b1"].update({k: b1[k] for k in ("wall_ms_per_step", "gpu_busy_ms_per_step",
                                                  "gpu_share_of_wall", "kernels_per_step")})
    return out


def gain_db(x, sr):
    """Linear gain to TARGET_LUFS, lowered if needed so the peak stays at PEAK_DBFS; boosts are
    capped at MAX_BOOST_DB so a near-silent take (one in the corpus) is not turned into loud noise."""
    import pyloudnorm
    lufs = pyloudnorm.Meter(sr).integrated_loudness(x)
    peak = 20 * np.log10(max(np.abs(x).max(), 1e-9))
    return min(TARGET_LUFS - lufs, PEAK_DBFS - peak, MAX_BOOST_DB)


def encode(src, dst, ffmpeg="ffmpeg"):
    from scipy.io import wavfile
    sr, w = wavfile.read(src)
    x = w.astype(np.float64) / (32768 if w.dtype == np.int16 else 1)
    g = gain_db(x if x.ndim == 1 else x.mean(axis=1), sr)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-af", f"volume={g:.2f}dB",
                    "-ac", "1", "-b:a", "64k", str(dst)], check=True)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data/corpus"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--wavs", default=str(ROOT / "data/wavs"))
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()
    recs = [json.loads(p.read_text()) for p in sorted(Path(a.corpus).glob("*.json"))]
    data = {"model": recs[0]["model"], "seconds": recs[0]["seconds"], "rounds": list(ROUNDS),
            "prompts": prompts(recs), "headline": headline(a.results)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"data.json: {len(data['prompts'])} prompts")
    if a.no_audio:
        return
    ff = shutil.which("ffmpeg") or SystemExit("ffmpeg not found (brew install ffmpeg)")
    (out / "audio").mkdir(exist_ok=True)
    gains = []
    for p in data["prompts"]:
        for row in range(p["n"]):
            dst = out / "audio" / f"{p['id']}_{row}.mp3"
            if not dst.exists():
                gains.append(encode(Path(a.wavs) / f"{p['id']}_{row}.wav", dst, ff))
    size = sum(f.stat().st_size for f in (out / "audio").glob("*.mp3")) / 1e6
    print(f"audio: {len(gains)} encoded, {size:.0f} MB total"
          + (f", gain {min(gains):+.1f} to {max(gains):+.1f} dB" if gains else ""))


if __name__ == "__main__":
    main()
