"""profile_decode.py trace arithmetic on a hand-made trace."""
import pytest
from profile_decode import busy_us, trace_stats


def test_busy_is_union_of_intervals():
    assert busy_us([(0, 10), (5, 15), (20, 25)]) == 20       # overlap counted once, gap excluded
    assert busy_us([]) == 0


def test_trace_stats_per_step():
    ev = [{"ph": "X", "cat": "kernel", "name": "gemm", "ts": 0, "dur": 600},
          {"ph": "X", "cat": "kernel", "name": "softmax", "ts": 1000, "dur": 200},
          {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH", "ts": 1100, "dur": 200},
          {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 0, "dur": 5},
          {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 900, "dur": 5},
          {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 50}]
    s = trace_stats({"traceEvents": ev}, steps=2)
    assert s["gpu_busy_ms_per_step"] == pytest.approx((600 + 300) / 1e3 / 2)   # 1000-1300 merged
    assert s["kernels_per_step"] == 1 and s["launch_calls_per_step"] == 1
    assert s["mean_kernel_us"] == 400 and s["top_kernels_share"][0] == ("gemm", 0.75)
