"""
Where does the pruning speedup come from? Two modes.

  python bench.py steps            # no downloads, ~minutes on an M1
  python bench.py run --prompts 20 # real musicgen-small + CLAP (~3.5 GB download)

`steps`: random weights with musicgen's real shapes (timing only needs shapes).
Times every decode step at batch 8/4/2/1, then compares
  - naive compute ratio      sum(8,8,8) / (8+4+2) = 1.71x  (what your note computed)
  - predicted wall-clock     from measured per-step time at each batch size
  - measured pruned run      the real loop incl. KV-cache slicing, cheap scorer
If per-step time barely drops with batch size (memory-bound decode), wall-clock
speedup << 1.71x. That is the number that decides whether "~5x" is explainable.

`run`: real model, CLAP scorer, per prompt: timed baseline vs timed pruned run,
plus the proxy-quality metrics the note lists as gaps (Spearman, false-kill,
winner CLAP delta). Row-id-keyed sampling (see prune.py) makes the pruned
winner one of the baseline's 8 candidates, so the quality comparison is exact.
"""
import argparse, json, math, time
import numpy as np
import mlx.core as mx
from prune import generate, random_model, undelay

SIZES = {"small": (1024, 24, 16, 4096), "medium": (1536, 48, 24, 6144)}  # HF configs


def segment_ms(clock, t0, cps):
    """Sum per-step wall time inside each [prev_cp, cp) segment, in ms."""
    dt = np.diff([t0] + clock) * 1e3
    edges = [0, *cps]
    return [float(dt[a:b].sum()) for a, b in zip(edges, edges[1:])]


def steps_mode(a):
    m = random_model(*SIZES[a.size])
    m.set_dtype(mx.float16 if a.dtype == "float16" else mx.float32)
    cond = mx.random.normal((1, 12, SIZES[a.size][0]))
    cps = [a.steps // 3, 2 * a.steps // 3, a.steps]
    generate(m, cond, 8, 50)                                  # warm up Metal kernels
    seg = {}
    for B in (8, 4, 2, 1):
        clock, t0 = [], time.perf_counter()
        generate(m, cond, B, a.steps, clock=clock)
        seg[B] = segment_ms(clock, t0, cps)
        print(f"batch {B} (x2 CFG = {2*B} rows): ms/step per third = "
              + ", ".join(f"{s/(a.steps//3):.1f}" for s in seg[B]))
    base = sum(seg[8])
    pred = seg[8][0] + seg[4][1] + seg[2][2]
    cheap = lambda f: f[..., 0].astype(mx.float32).mean(axis=1)  # ~free scorer
    clock, t0 = [], time.perf_counter()
    _, ids, _ = generate(m, cond, 8, a.steps, cheap, cps, clock=clock)
    meas = time.perf_counter() - t0
    out = {"size": a.size, "dtype": a.dtype, "steps": a.steps,
           "naive_compute_x": 24 / 14, "predicted_wallclock_x": base / pred,
           "measured_wallclock_x": base / 1e3 / meas, "baseline_s": base / 1e3,
           "pruned_s": meas, "ms_per_step": {B: [s / (a.steps // 3) for s in v] for B, v in seg.items()}}
    print(json.dumps({k: v for k, v in out.items() if k != "ms_per_step"}, indent=1))
    assert len(ids) == 1 and seg[8][0] > 0
    json.dump(out, open(f"results_steps_{a.size}_{a.dtype}.json", "w"), indent=1)
    print("OK: speedup decomposed into per-step batch scaling")


class Clap:
    """CLAP text-audio cosine similarity (PyTorch, CPU). The note's scorer."""
    def __init__(self, name="laion/clap-htsat-unfused"):
        from transformers import ClapModel, ClapProcessor
        self.model, self.proc = ClapModel.from_pretrained(name).eval(), ClapProcessor.from_pretrained(name)

    def _feat(self, out):  # transformers 5 may return a ModelOutput instead of a tensor
        return out if hasattr(out, "norm") else out.pooler_output

    def text(self, prompt):
        import torch
        with torch.no_grad():
            t = self._feat(self.model.get_text_features(**self.proc.tokenizer([prompt], return_tensors="pt")))
        self.t = t / t.norm(dim=-1, keepdim=True)

    def __call__(self, wavs_32k):  # (B, samples) numpy
        """Unfused CLAP sees 10s max, and its processor RANDOMLY crops longer clips
        ("rand_trunc") -- a 30s checkpoint score would be a random 10s window, and
        two runs could prune different rows. So: cut each clip into 10s chunks,
        embed every chunk, average per clip. Deterministic, and covers the whole clip."""
        import torch
        from scipy.signal import resample_poly
        W = 480_000                                                   # 10 s at 48 kHz
        chunks, owner = [], []
        for b, w in enumerate(wavs_32k):
            w = resample_poly(w, 3, 2).astype(np.float32)             # CLAP wants 48 kHz
            for s in range(0, max(len(w) - W // 10, 1), W):           # drop tails < 1 s
                chunks.append(w[s : s + W]); owner.append(b)
        x = self.proc.feature_extractor(chunks, sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            a = self._feat(self.model.get_audio_features(**x))
        a = a / a.norm(dim=-1, keepdim=True)
        owner = torch.tensor(owner)
        clip = torch.stack([a[owner == b].mean(0) for b in range(len(wavs_32k))])
        return ((clip / clip.norm(dim=-1, keepdim=True)) @ self.t.T)[:, 0].numpy()


def run_mode(a):
    import sys; sys.path.insert(0, "vendor")
    from musicgen import MusicGen
    from scipy.stats import spearmanr
    m, clap = MusicGen.from_pretrained(f"facebook/musicgen-{a.size}"), Clap()
    K, S = m.num_codebooks, a.steps
    cps = [S // 3, 2 * S // 3, S]

    def score(frames):  # decode clip-so-far -> waveform -> CLAP vs prompt
        wav = m._audio_decoder.decode(mx.swapaxes(frames, -1, -2)[:, None], audio_scales=[None])
        return mx.array(clap(np.array(wav.astype(mx.float32)).reshape(frames.shape[0], -1)))

    prompts = [l.strip() for l in open(a.prompt_file) if l.strip()][: a.prompts]
    clap.text(prompts[0]); generate(m, m.text_conditioner(prompts[0]), 8, 30)  # warm-up
    rows = []
    for i, p in enumerate(prompts):
        clap.text(p); cond = m.text_conditioner(p)
        t0 = time.perf_counter()                                  # baseline: 8 full, score at end
        base, _, _ = generate(m, cond, 8, S, seed=i)
        final = np.array(score(undelay(base, K)))
        t_base = time.perf_counter() - t0
        t0 = time.perf_counter()                                  # pruned: 8 -> 4 -> 2 -> 1
        _, ids, log = generate(m, cond, 8, S, score, cps, seed=i)
        t_prune = time.perf_counter() - t0
        # untimed: checkpoint scores for ALL 8 baseline rows (valid: prefixes don't change)
        cp = {c: np.array(score(undelay(base[:, : c + 1], K))) for c in cps[:-1]}
        top2 = set(np.argsort(-final)[:2].tolist())
        rows.append({
            "prompt": p, "speedup_x": t_base / t_prune, "t_base": t_base, "t_prune": t_prune,
            "spearman": {c: float(spearmanr(cp[c], final)[0]) for c in cp},
            "false_kill": len(top2 - set(log[cps[-1]])) / 2,    # top-2 finals that never reached the end
            "same_winner": int(ids[0] == int(final.argmax())),
            "clap_loss": float(final.max() - final[ids[0]]),    # >= 0 by construction
        })
        assert abs(log[cps[0]][0] - cp[cps[0]][0]) < 1e-3       # pruned run saw the same row 0
        print(json.dumps(rows[-1]))
    summ = {k: float(np.median([r[k] for r in rows])) for k in ("speedup_x", "false_kill", "same_winner", "clap_loss")}
    summ["same_winner"] = float(np.mean([r["same_winner"] for r in rows]))
    print("MEDIAN over prompts:", json.dumps(summ))
    json.dump({"summary": summ, "rows": rows}, open(f"results_run_{a.size}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["steps", "run"])
    ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--steps", type=int, default=1500)  # 1500 frames = 30s = one context window
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--prompt-file", default="prompts.txt")
    a = ap.parse_args()
    steps_mode(a) if a.mode == "steps" else run_mode(a)
