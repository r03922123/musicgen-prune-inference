"""
Where does a decode step's time go? torch.profiler over 50 decode steps at batch 1 and 8
Tests the hypothesis that the ~18 ms/step floor seen on the T4 at batch <= 4
(results_steps_torch_small_float16_pretrained_t4.json) is CPU-side launch overhead, not GPU work.

  python torch/profile_decode.py                 # real musicgen-small, fp16, CUDA
  python torch/profile_decode.py --random        # random weights, no download (CPU ok, smoke only)

Per batch size: an unprofiled timed run (wall ms/step), then a profiled run of the same steps.
From the profiler's trace: GPU busy time = union of kernel + memcpy/memset intervals on the device;
GPU share = busy / unprofiled wall; the rest of the wall time is time the GPU sits idle waiting
for the CPU (Python, HF module dispatch, kernel launches). Writes results/profile_decode_<tag>.json.
"""
import argparse, json, os, tempfile, time
from collections import Counter
from pathlib import Path
import torch
from bench import load
from prune import generate

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def busy_us(intervals):
    """Total length of the union of [start, end) intervals."""
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if e > end:
            total += e - max(s, end)
            end = e
    return total


def trace_stats(trace, steps):
    ev = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    gpu = [e for e in ev if e.get("cat") in GPU_CATS]
    kern = [e for e in gpu if e["cat"] == "kernel"]
    launches = [e for e in ev if e.get("cat") == "cuda_runtime" and "Launch" in e.get("name", "")]
    sync = [e for e in ev if e.get("cat") == "cuda_runtime" and "Synchronize" in e.get("name", "")]
    by_name = Counter()
    for e in kern:
        by_name[e["name"][:90]] += e["dur"]
    ktot = sum(e["dur"] for e in kern) or 1.0
    return {"gpu_busy_ms_per_step": busy_us([(e["ts"], e["ts"] + e["dur"]) for e in gpu]) / 1e3 / steps,
            "kernels_per_step": len(kern) / steps,
            "mean_kernel_us": ktot / max(len(kern), 1),
            "launch_calls_per_step": len(launches) / steps,
            "launch_cpu_ms_per_step": sum(e["dur"] for e in launches) / 1e3 / steps,
            "sync_wait_ms_per_step": sum(e["dur"] for e in sync) / 1e3 / steps,
            "top_kernels_share": [(n, round(d / ktot, 3)) for n, d in by_name.most_common(8)]}


def timed(m, cond, B, steps, dev):
    clock, t0 = [], time.perf_counter()
    generate(m, cond, B, steps, clock=clock)
    return (clock[-1] - t0) * 1e3 / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, cond = load(a, dev, getattr(torch, a.dtype))
    gpu = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    res = {"device": gpu, "size": a.size, "dtype": a.dtype, "weights": "random" if a.random else "pretrained",
           "steps": a.steps, "torch": torch.__version__, "batches": {}}
    generate(m, cond, 8, 50)                                         # warm up kernels / allocator
    for B in a.batches:
        wall = min(timed(m, cond, B, a.steps, dev) for _ in range(3))  # best of 3, unprofiled
        with torch.profiler.profile(activities=acts) as prof:
            clock, t0 = [], time.perf_counter()
            generate(m, cond, B, a.steps, clock=clock)
        wall_prof = (clock[-1] - t0) * 1e3 / a.steps
        with tempfile.TemporaryDirectory() as d:
            prof.export_chrome_trace(os.path.join(d, "t.json"))
            st = trace_stats(json.loads(Path(d, "t.json").read_text()), a.steps)
        cpu_top = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:8]
        st.update({"wall_ms_per_step": wall, "wall_ms_per_step_profiled": wall_prof,
                   "gpu_share_of_wall": st["gpu_busy_ms_per_step"] / wall,
                   "gpu_idle_ms_per_step": wall - st["gpu_busy_ms_per_step"],
                   "top_cpu_ops_self_ms_per_step": [(e.key, round(e.self_cpu_time_total / 1e3 / a.steps, 3))
                                                    for e in cpu_top]})
        res["batches"][str(B)] = st
        print(f"batch {B} (x2 CFG = {2*B} rows): wall {wall:.1f} ms/step, GPU busy "
              f"{st['gpu_busy_ms_per_step']:.1f} ms/step ({st['gpu_share_of_wall']:.0%}), "
              f"{st['kernels_per_step']:.0f} kernels/step, mean kernel {st['mean_kernel_us']:.0f} us, "
              f"launch API {st['launch_cpu_ms_per_step']:.1f} ms/step", flush=True)
    tag = f"{a.size}_{a.dtype}_{res['weights']}_{gpu.split()[-1].lower()}"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"profile_decode_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
