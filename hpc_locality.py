#!/usr/bin/env python3
"""
HPC Data-Structure Optimization Benchmarks
------------------------------------------
Focus: Data locality & layout (AoS -> SoA), strided vs contiguous access,
and cache-friendly blocking for matrix multiply.

This aligns with common fixes in the MSR'23 empirical study of HPC performance bugs
(e.g., data-locality optimizations, micro-architecture aware changes, and guiding compilers).

USAGE (examples):
  python hpc_locality.py
  python hpc_locality.py --n-dot 600000 --repeats 5
  python hpc_locality.py --no-plots
  python hpc_locality.py --mat-n 192 --mat-repeats 3
  python hpc_locality.py --force-py-mm   # force pure-Python mm (can be slow)
  python hpc_locality.py --no-numba      # ignore numba even if installed

Outputs:
  - CSV:   results_hpc_locality_<timestamp>.csv
  - PNGs:  dot_bench_<timestamp>.png, stride_bench_<timestamp>.png, mm_bench_<timestamp>.png (if plotting enabled)
"""

import argparse
import csv
import math
import os
import platform
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Tuple

# Optional deps
try:
    import numpy as np
except Exception as e:
    print("ERROR: This script requires NumPy. Please install with `pip install numpy`.", file=sys.stderr)
    raise

try:
    import psutil
except Exception:
    psutil = None

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Optional numba
HAVE_NUMBA = False
if "--no-numba" not in sys.argv:
    try:
        from numba import njit, prange
        HAVE_NUMBA = True
    except Exception:
        HAVE_NUMBA = False

# ----------------------------
# Helpers
# ----------------------------

def secs_fmt(x: float) -> str:
    if x < 1e-3:
        return f"{x*1e6:.1f} µs"
    if x < 1:
        return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"

def time_once(fn: Callable, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0

def time_repeat(fn: Callable, repeats: int, warmup: int = 1, *args, **kwargs) -> Tuple[float, float, List[float]]:
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    # Timed runs
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*args, **kwargs)
        samples.append(time.perf_counter() - t)
    mean = sum(samples) / len(samples)
    # simple stdev
    var = sum((s - mean)**2 for s in samples) / (len(samples) - 1) if len(samples) > 1 else 0.0
    stdev = math.sqrt(var)
    return mean, stdev, samples

def system_info() -> Dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": str(os.cpu_count()),
    }
    if psutil:
        try:
            info["cpu_count_physical"] = str(psutil.cpu_count(logical=False))
            info["memory_gb"] = f"{psutil.virtual_memory().total / 1e9:.1f}"
        except Exception:
            pass
    return info

# ----------------------------
# Bench 1: AoS vs SoA dot products
# ----------------------------

def gen_dot_data(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    A = rng.random((n, 3), dtype=np.float64)
    B = rng.random((n, 3), dtype=np.float64)
    # AoS: list of tuples; SoA: separate arrays
    aos = [(float(A[i,0]), float(A[i,1]), float(A[i,2]),
            float(B[i,0]), float(B[i,1]), float(B[i,2])) for i in range(n)]
    return A, B, aos

def dot_aos_py(aos) -> float:
    s = 0.0
    for (ax, ay, az, bx, by, bz) in aos:
        s += ax*bx + ay*by + az*bz
    return s

def dot_soa_numpy(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sum(A * B))

if HAVE_NUMBA:
    @njit(fastmath=True)
    def dot_aos_numba(aos_arr: np.ndarray) -> float:
        s = 0.0
        # aos_arr shape: (n, 6)
        for i in range(aos_arr.shape[0]):
            ax, ay, az, bx, by, bz = aos_arr[i, 0], aos_arr[i, 1], aos_arr[i, 2], aos_arr[i, 3], aos_arr[i, 4], aos_arr[i, 5]
            s += ax*bx + ay*by + az*bz
        return s

    @njit(fastmath=True, parallel=True)
    def dot_soa_numba(A: np.ndarray, B: np.ndarray) -> float:
        s = 0.0
        for i in prange(A.shape[0]):
            s += A[i,0]*B[i,0] + A[i,1]*B[i,1] + A[i,2]*B[i,2]
        return s

def run_dot_bench(n: int, repeats: int, use_numba: bool) -> List[Dict]:
    A, B, aos = gen_dot_data(n)
    results = []

    # Verify equivalence
    ref = dot_soa_numpy(A, B)
    # AoS Python
    m, s, _ = time_repeat(dot_aos_py, repeats, 1, aos)
    results.append({"bench": "Dot AoS (Python loop)", "N": n, "mean_s": m, "stdev_s": s})
    # SoA NumPy
    m, s, _ = time_repeat(dot_soa_numpy, repeats, 1, A, B)
    results.append({"bench": "Dot SoA (NumPy vectorized)", "N": n, "mean_s": m, "stdev_s": s})

    if use_numba and HAVE_NUMBA:
        # Prepare contiguous AoS array for Numba
        aos_arr = np.asarray(aos, dtype=np.float64).reshape(-1, 6)
        # compile
        dot_aos_numba(aos_arr)
        dot_soa_numba(A, B)
        # timed
        m, s, _ = time_repeat(dot_aos_numba, repeats, 1, aos_arr)
        results.append({"bench": "Dot AoS (Numba JIT)", "N": n, "mean_s": m, "stdev_s": s})

        m, s, _ = time_repeat(dot_soa_numba, repeats, 1, A, B)
        results.append({"bench": "Dot SoA (Numba JIT, parallel)", "N": n, "mean_s": m, "stdev_s": s})

    # quick sanity check
    assert abs(ref - dot_soa_numpy(A, B)) < 1e-6, "Ref mismatch"
    return results

# ----------------------------
# Bench 2: Strided vs Contiguous reductions
# ----------------------------

def gen_stride_data(n: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    X = rng.random(n, dtype=np.float64)
    return X

def sum_strided(X: np.ndarray, stride: int) -> float:
    return float(np.sum(X[::stride]))

def sum_contiguous_copy(X: np.ndarray, stride: int) -> float:
    # materialize to contiguous buffer, then reduce
    view = X[::stride]
    contig = np.ascontiguousarray(view)
    return float(np.sum(contig))

def sum_chunked(X: np.ndarray, chunk: int) -> float:
    # reduce in cache-friendly chunks
    s = 0.0
    for i in range(0, X.size, chunk):
        s += float(np.sum(X[i:i+chunk]))
    return s

def run_stride_bench(n: int, stride: int, chunk: int, repeats: int) -> List[Dict]:
    X = gen_stride_data(n)
    results = []

    # Strided sum
    m, s, _ = time_repeat(sum_strided, repeats, 1, X, stride)
    results.append({"bench": f"Strided sum (stride={stride})", "N": n, "mean_s": m, "stdev_s": s})

    # Contiguous copy then sum
    m, s, _ = time_repeat(sum_contiguous_copy, repeats, 1, X, stride)
    results.append({"bench": f"Contiguous copy+sum (stride={stride})", "N": n, "mean_s": m, "stdev_s": s})

    # Chunked sum (no stride) to show simple blocking
    m, s, _ = time_repeat(sum_chunked, repeats, 1, X, chunk)
    results.append({"bench": f"Chunked sum (chunk={chunk})", "N": n, "mean_s": m, "stdev_s": s})

    # Full contiguous baseline
    m, s, _ = time_repeat(np.sum, repeats, 1, X)
    results.append({"bench": "Contiguous full sum", "N": n, "mean_s": float(m), "stdev_s": float(s)})

    return results

# ----------------------------
# Bench 3: Matrix multiply (naive vs blocked vs NumPy)
# ----------------------------

def gen_mm_data(n: int, seed: int = 99):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n), dtype=np.float64)
    B = rng.random((n, n), dtype=np.float64)
    return A, B

def mm_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B

def mm_naive_py(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # WARNING: very slow in pure Python for large N
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(n):
            aik = A[i, k]
            for j in range(n):
                C[i, j] += aik * B[k, j]
    return C

def mm_blocked_py(A: np.ndarray, B: np.ndarray, bs: int = 32) -> np.ndarray:
    # Still Python loops; uses blocking to improve locality
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for ii in range(0, n, bs):
        for kk in range(0, n, bs):
            for jj in range(0, n, bs):
                i_max = min(ii+bs, n)
                k_max = min(kk+bs, n)
                j_max = min(jj+bs, n)
                for i in range(ii, i_max):
                    for k in range(kk, k_max):
                        aik = A[i, k]
                        for j in range(jj, j_max):
                            C[i, j] += aik * B[k, j]
    return C

if HAVE_NUMBA:
    @njit(fastmath=True)
    def mm_naive_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for k in range(n):
                aik = A[i, k]
                for j in range(n):
                    C[i, j] += aik * B[k, j]
        return C

    @njit(fastmath=True, parallel=True)
    def mm_blocked_numba(A: np.ndarray, B: np.ndarray, bs: int) -> np.ndarray:
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        for ii in prange(0, n, bs):
            for kk in range(0, n, bs):
                for jj in range(0, n, bs):
                    i_max = min(ii+bs, n)
                    k_max = min(kk+bs, n)
                    j_max = min(jj+bs, n)
                    for i in range(ii, i_max):
                        for k in range(kk, k_max):
                            aik = A[i, k]
                            for j in range(jj, j_max):
                                C[i, j] += aik * B[k, j]
        return C

def run_mm_bench(n: int, repeats: int, block: int, force_py: bool, use_numba: bool) -> List[Dict]:
    A, B = gen_mm_data(n)
    results = []

    # NumPy BLAS baseline
    ref = mm_numpy(A, B)
    m, s, _ = time_repeat(mm_numpy, repeats, 1, A, B)
    results.append({"bench": "MatMul NumPy (BLAS)", "N": n, "mean_s": m, "stdev_s": s})

    # Pure Python loops are extremely slow; only run if forced or if numba absent and n is small
    if force_py:
        # naive
        m, s, _ = time_repeat(mm_naive_py, repeats, 0, A, B)
        C1 = mm_naive_py(A, B)
        assert np.allclose(C1, ref, atol=1e-8)
        results.append({"bench": "MatMul naive (Python loops)", "N": n, "mean_s": m, "stdev_s": s})
        # blocked
        m, s, _ = time_repeat(mm_blocked_py, repeats, 0, A, B, block)
        C2 = mm_blocked_py(A, B, block)
        assert np.allclose(C2, ref, atol=1e-8)
        results.append({"bench": f"MatMul blocked (Python loops, bs={block})", "N": n, "mean_s": m, "stdev_s": s})

    if use_numba and HAVE_NUMBA:
        # compile first
        mm_naive_numba(A, B)
        mm_blocked_numba(A, B, block)
        # naive numba
        m, s, _ = time_repeat(mm_naive_numba, repeats, 0, A, B)
        C3 = mm_naive_numba(A, B)
        assert np.allclose(C3, ref, atol=1e-8)
        results.append({"bench": "MatMul naive (Numba JIT)", "N": n, "mean_s": m, "stdev_s": s})
        # blocked numba
        m, s, _ = time_repeat(mm_blocked_numba, repeats, 0, A, B, block)
        C4 = mm_blocked_numba(A, B, block)
        assert np.allclose(C4, ref, atol=1e-8)
        results.append({"bench": f"MatMul blocked (Numba JIT, bs={block})", "N": n, "mean_s": m, "stdev_s": s})

    return results

# ----------------------------
# Plotting
# ----------------------------

def save_barplot(rows: List[Dict], title: str, fname: str):
    if not HAVE_MPL:
        return
    labels = [r["bench"] for r in rows]
    vals = [r["mean_s"] for r in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, vals)
    plt.ylabel("Seconds (mean)")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="HPC data locality/layout benchmarks")
    parser.add_argument("--n-dot", type=int, default=300_000, help="N for dot-product AoS vs SoA")
    parser.add_argument("--n-stride", type=int, default=4_000_000, help="N for strided/contiguous reductions")
    parser.add_argument("--stride", type=int, default=8, help="Stride for strided reduction")
    parser.add_argument("--chunk", type=int, default=131072, help="Chunk size for chunked reduction")
    parser.add_argument("--mat-n", type=int, default=192, help="Matrix size N (NxN) for mm benchmark")
    parser.add_argument("--block", type=int, default=32, help="Block size for blocked matmul")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per benchmark")
    parser.add_argument("--mat-repeats", type=int, default=1, help="Repeats per matrix multiply variant")
    parser.add_argument("--no-plots", action="store_true", help="Disable PNG plots")
    parser.add_argument("--no-numba", action="store_true", help="Ignore Numba even if installed")
    parser.add_argument("--force-py-mm", action="store_true", help="Force pure-Python matmul loops (slow)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"results_hpc_locality_{timestamp}.csv"
    out_rows: List[Dict] = []

    # Header: system info
    info = system_info()
    print("\n=== System Info ===")
    for k, v in info.items():
        print(f"{k:>20}: {v}")

    print("\n=== Bench 1: Dot Products (AoS vs SoA) ===")
    dot_rows = run_dot_bench(args.n_dot, args.repeats, use_numba=(not args.no_numba))
    for r in dot_rows:
        print(f"{r['bench']:<36} N={r['N']:<8} mean={secs_fmt(r['mean_s'])}  stdev={secs_fmt(r['stdev_s'])}")
    out_rows.extend([{"group":"dot", **r} for r in dot_rows])

    print("\n=== Bench 2: Strided vs Contiguous ===")
    stride_rows = run_stride_bench(args.n_stride, args.stride, args.chunk, args.repeats)
    for r in stride_rows:
        print(f"{r['bench']:<36} N={r['N']:<8} mean={secs_fmt(r['mean_s'])}  stdev={secs_fmt(r['stdev_s'])}")
    out_rows.extend([{"group":"stride", **r} for r in stride_rows])

    print("\n=== Bench 3: Matrix Multiply (naive/blocked/NumPy) ===")
    mm_rows = run_mm_bench(args.mat_n, args.mat_repeats, args.block, args.force_py_mm, use_numba=(not args.no_numba))
    for r in mm_rows:
        print(f"{r['bench']:<36} N={r['N']:<8} mean={secs_fmt(r['mean_s'])}  stdev={secs_fmt(r['stdev_s'])}")
    out_rows.extend([{"group":"mm", **r} for r in mm_rows])

    # Save CSV
    with open(csv_name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "bench", "N", "mean_s", "stdev_s"])
        for r in out_rows:
            w.writerow([r.get("group",""), r["bench"], r["N"], f"{r['mean_s']:.9f}", f"{r['stdev_s']:.9f}"])
    print(f"\nSaved CSV: {csv_name}")

    # Save plots
    if not args.no_plots and HAVE_MPL:
        save_barplot(dot_rows, f"Dot Products AoS vs SoA (N={args.n_dot})", f"dot_bench_{timestamp}.png")
        save_barplot(stride_rows, f"Strided vs Contiguous (N={args.n_stride}, stride={args.stride})", f"stride_bench_{timestamp}.png")
        if mm_rows:
            save_barplot(mm_rows, f"Matrix Multiply (N={args.mat_n})", f"mm_bench_{timestamp}.png")
        print("Saved plots (PNG).")
    elif args.no_plots:
        print("Plotting disabled (--no-plots).")
    else:
        print("matplotlib not available; skipping plots.")


if __name__ == "__main__":
    main()
