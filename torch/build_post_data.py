"""
Build the post's interactive-figure data and audio (post/). Mac, no GPU, no model calls.

  python torch/build_post_data.py                 # post/data.json + post/audio/<id>_<row>.mp3
  python torch/build_post_data.py --no-audio      # data.json only

data.json holds, per prompt, the caption and the CLAP scores of every 5 s prefix. The pruning figure recomputes the committed results from it in the browser; post/tests/sim.test.mjs checks they match results/. Audio: every
take loudness-normalised to -16 LUFS with one linear gain (no compression, so slow builds keep
their dynamics), peak-limited, boost capped at +15 dB, 64 kbps mono MP3.
"""
import argparse, json, re, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = (5, 10, 15, 20, 25, 30)
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
                    "clap": {str(t): s["clap"][str(t)] for t in PREFIXES}})
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
    ap.add_argument("--wavs", default=str(ROOT / "data/wavs"))
    ap.add_argument("--out", default=str(ROOT / "post"))
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()
    recs = [json.loads(p.read_text()) for p in sorted(Path(a.corpus).glob("*.json"))]
    data = {"model": recs[0]["model"], "seconds": recs[0]["seconds"], "prefixes": list(PREFIXES),
            "prompts": prompts(recs)}
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
