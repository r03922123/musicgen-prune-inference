"""
Where does the pruning speedup come from? PyTorch/CUDA port of ../bench.py `steps`.

  python torch/bench.py steps            # real facebook/musicgen-small decoder (~2.4 GB download)
  python torch/bench.py steps --random   # random weights, musicgen-small shapes, no downloads

Times every decode step at batch 8/4/2/1, then compares
  - naive compute ratio      sum(8,8,8) / (8+4+2) = 1.71x
  - predicted wall-clock     from measured per-step time at each batch size
  - measured pruned run      the real loop incl. KV-cache slicing, cheap scorer
Also reports whether the pruned survivor matches the same row of the batch-8 run
token-for-token at full scale: on GPU, kernels picked for 16 vs 8 rows may round
differently, so this is measured rather than assumed.
"""
import argparse, json, time
import numpy as np
import torch
from prune import generate, random_model, undelay, toy_score

SIZES = {"small": (1024, 24, 16, 4096), "medium": (1536, 48, 24, 6144)}  # HF configs
PROMPT = "80s pop track with bassy drums and synth"


def segment_ms(clock, t0, cps):
    """Sum per-step wall time inside each [prev_cp, cp) segment, in ms."""
    dt = np.diff([t0] + clock) * 1e3
    edges = [0, *cps]
    return [float(dt[a:b].sum()) for a, b in zip(edges, edges[1:])]


def load(a, dev, dtype):
    """-> (decoder MusicgenForCausalLM, cond (1, T, D))."""
    if a.random:
        m = random_model(*SIZES[a.size])
        torch.manual_seed(0)
        return m.to(dev, dtype), torch.randn(1, 12, SIZES[a.size][0])
    from generate_corpus import load_musicgen, text_cond     # T5 stays fp32; only the decoder is cast
    full, proc = load_musicgen(a.size, dev, dtype)
    return full.decoder, text_cond(full, proc, PROMPT)


def steps_mode(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if a.dtype == "float16" else torch.float32
    m, cond = load(a, dev, dtype)
    cps = [a.steps // 3, 2 * a.steps // 3, a.steps]
    gpu = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    print(f"device={gpu} weights={'random' if a.random else 'pretrained'} dtype={a.dtype} steps={a.steps}")
    generate(m, cond, 8, 50)                                  # warm up kernels / allocator
    seg = {}
    for B in (8, 4, 2, 1):
        clock, t0 = [], time.perf_counter()
        out, _, _ = generate(m, cond, B, a.steps, clock=clock)
        if B == 8:
            base8 = out
        seg[B] = segment_ms(clock, t0, cps)
        print(f"batch {B} (x2 CFG = {2*B} rows): ms/step per third = "
              + ", ".join(f"{s/(a.steps//3):.1f}" for s in seg[B]), flush=True)
    base = sum(seg[8])
    pred = seg[8][0] + seg[4][1] + seg[2][2]
    clock, t0 = [], time.perf_counter()
    pr, ids, log = generate(m, cond, 8, a.steps, toy_score, cps, clock=clock)
    meas = time.perf_counter() - t0
    K = m.config.num_codebooks
    eq = (undelay(pr, K) == undelay(base8, K)[ids]).all(dim=-1)[0]   # (T,) frame-level match
    first_diff = int((~eq).nonzero()[0]) if not eq.all() else None
    out = {"size": a.size, "dtype": a.dtype, "steps": a.steps, "device": gpu,
           "weights": "random" if a.random else "pretrained",
           "torch": torch.__version__, "transformers": __import__("transformers").__version__,
           "naive_compute_x": 24 / 14, "predicted_wallclock_x": base / pred,
           "measured_wallclock_x": base / 1e3 / meas, "baseline_s": base / 1e3,
           "pruned_s": meas, "survivor": ids[0],
           "survivor_frame_match": float(eq.float().mean()), "survivor_first_diff_frame": first_diff,
           "ms_per_step": {B: [s / (a.steps // 3) for s in v] for B, v in seg.items()}}
    print(json.dumps({k: v for k, v in out.items() if k != "ms_per_step"}, indent=1))
    assert len(ids) == 1 and seg[8][0] > 0
    tag = "random" if a.random else "pretrained"
    json.dump(out, open(f"results_steps_torch_{a.size}_{a.dtype}_{tag}.json", "w"), indent=1)
    print("OK: speedup decomposed into per-step batch scaling")


if __name__ == "__main__":
    from transformers.utils import logging
    logging.set_verbosity_error()
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["steps"])
    ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--steps", type=int, default=1500)  # 1500 frames = 30s = one context window
    ap.add_argument("--random", action="store_true", help="random weights, no downloads")
    steps_mode(ap.parse_args())
