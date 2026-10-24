"""
Blind A/B listening test (the human check of Q2), macOS (plays with afplay).

  uv run python torch/listen.py                 # 20 trials, resumes from results/listening.csv
  uv run python torch/listen.py --analyse       # binomial test on the CSV only

Trial = one MusicCaps prompt (20 of 40, seed 0): the unpruned CLAP best-of-8 pick vs one
random other candidate of the same prompt, in random A/B order. Both clips are loudness
matched (same RMS) so "louder" cannot win. You see the caption, hear A then B, and choose.
The CSV records which side was the pick only after you answer; do not open it mid-test.

Needs data/corpus/<id>.json (committed) and data/wavs/<id>_<row>.wav (from the corpus
kernel's wavs.tar). Analysis: two-sided exact binomial test of "CLAP pick preferred" vs 0.5.
With 20 trials p < 0.05 needs >= 15/20 (75 %) one way, so only large effects are detectable.
"""
import argparse, csv, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["trial", "id", "A", "B", "pick", "other", "choice", "chose_pick", "replays", "seconds", "time"]


def make_plan(corpus, n=20, seed=0):
    """Deterministic, blind trial list: [{trial, id, caption, pick, other, A, B}]."""
    recs = [json.loads(p.read_text()) for p in sorted(Path(corpus).glob("mc_*.json"))]
    rng = np.random.default_rng(seed)
    out = []
    for i, k in enumerate(sorted(rng.choice(len(recs), n, replace=False))):
        r = recs[k]
        clap = r["scores"]["clap"]["30"]
        pick = int(np.argmax(clap))
        other = int(rng.choice([j for j in range(r["n"]) if j != pick]))
        a, b = (pick, other) if rng.random() < 0.5 else (other, pick)
        out.append({"trial": i, "id": r["id"], "caption": r["prompt"], "pick": pick, "other": other,
                    "A": a, "B": b})
    return out


def analyse(rows):
    """Two-sided exact binomial test, H0: P(choose CLAP pick) = 0.5; Clopper-Pearson 95% CI."""
    from scipy.stats import binomtest
    k, n = sum(r["chose_pick"] == "1" for r in rows), len(rows)
    t = binomtest(k, n, 0.5)
    ci = t.proportion_ci(0.95)
    return {"n": n, "k_pick": k, "share_pick": k / n if n else float("nan"),
            "share_lo": ci.low, "share_hi": ci.high, "p_two_sided": t.pvalue}


def matched(src, dst, rms=0.05):
    """Write src rescaled to a common RMS (peak-limited to 0.99) so loudness cannot decide."""
    from scipy.io import wavfile
    sr, w = wavfile.read(src)
    x = w.astype(np.float32) / 32767
    x *= min(rms / max(np.sqrt((x ** 2).mean()), 1e-6), 0.99 / max(np.abs(x).max(), 1e-6))
    wavfile.write(dst, sr, (x * 32767).astype(np.int16))


def play(path, seconds):
    try:
        subprocess.run(["afplay", "-t", str(seconds), str(path)], check=False)
    except KeyboardInterrupt:          # Ctrl-C skips the rest of a clip, not the test
        pass


def run(a):
    out = Path(a.csv)
    done = list(csv.DictReader(out.open())) if out.exists() else []
    plan = make_plan(a.corpus, a.n, a.seed)
    missing = [f"{t['id']}_{r}.wav" for t in plan for r in (t["A"], t["B"])
               if not (Path(a.wavs) / f"{t['id']}_{r}.wav").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} WAVs missing from {a.wavs}, e.g. {missing[0]}")
    todo = [t for t in plan if t["trial"] >= len(done)]
    print(f"{len(done)} of {len(plan)} trials done; {len(todo)} to go. Keys: 1 = choose A, 2 = choose B, "
          "a / b = replay A / B, r = replay both. Ctrl-C during a clip skips the rest of it.\n")
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", newline="") as f, tempfile.TemporaryDirectory() as tmp:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for t in todo:
            clips = {}
            for side in "AB":
                clips[side] = Path(tmp) / f"{side}.wav"
                matched(Path(a.wavs) / f"{t['id']}_{t[side]}.wav", clips[side])
            print(f"--- trial {t['trial'] + 1}/{len(plan)} ---\nCaption: {t['caption']}\n")
            t0, replays, todo_play = time.time(), 0, "AB"
            while True:
                for side in todo_play:
                    print(f"  playing {side} ...", flush=True)
                    play(clips[side], a.seconds)
                key = input("Which would you rather keep for this caption? [1=A, 2=B, a, b, r]: ").strip().lower()
                if key in ("1", "2"):
                    break
                todo_play = {"a": "A", "b": "B", "r": "AB"}.get(key, "")
                replays += bool(todo_play)
            choice = "A" if key == "1" else "B"
            w.writerow({"trial": t["trial"], "id": t["id"], "A": t["A"], "B": t["B"], "pick": t["pick"],
                        "other": t["other"], "choice": choice, "chose_pick": int(t[choice] == t["pick"]),
                        "replays": replays, "seconds": round(time.time() - t0, 1),
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            f.flush()
            print()
    report(out)


def report(path):
    rows = list(csv.DictReader(Path(path).open()))
    r = analyse(rows)
    print(f"CLAP pick preferred in {r['k_pick']}/{r['n']} trials ({r['share_pick']:.0%}, 95% CI "
          f"{r['share_lo']:.0%}-{r['share_hi']:.0%}); two-sided binomial p = {r['p_two_sided']:.3f}")
    Path(path).with_suffix(".json").write_text(json.dumps(r, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data/corpus"))
    ap.add_argument("--wavs", default=str(ROOT / "data/wavs"))
    ap.add_argument("--csv", default=str(ROOT / "results/listening.csv"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=30, help="play at most this many seconds per clip")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    report(a.csv) if a.analyse else run(a)


if __name__ == "__main__":
    main()
