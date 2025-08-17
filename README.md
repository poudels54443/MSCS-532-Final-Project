# HPC Data Locality & Layout Optimization (AoS → SoA)

## Overview
This project demonstrates a practical optimization emphasized in the MSR’23 empirical study on HPC performance bugs: improving **data locality and memory layout**. The benchmark suite shows how moving from an **Array‑of‑Structs (AoS)** representation to a **Structure‑of‑Arrays (SoA)** layout, using **unit‑stride traversal**, and applying **simple blocking** changes performance for common kernels. The code is a small, reproducible prototype intended for coursework and lab write‑ups.

## How to Run
1. Ensure Python and NumPy are installed (Matplotlib is optional for charts; Numba is optional for JIT paths).
   ```bash
   pip install numpy matplotlib numba psutil
   ```
2. Run the benchmark script with defaults:
   ```bash
   python hpc_locality_benchmarks.py
   ```
   This prints a summary, writes a CSV named like `results_hpc_locality_YYYYMMDD_HHMMSS.csv`, and saves PNG charts if Matplotlib is available.

## Console Output (from one run)
```text
(myenv) safal@Safals-MBP Operating System % python3 hpc_locality.py

=== System Info ===
              python: 3.7.16
               numpy: 1.21.6
            platform: Darwin-23.4.0-arm64-arm-64bit
           processor: arm
   cpu_count_logical: 8
  cpu_count_physical: 8
           memory_gb: 17.2

=== Bench 1: Dot Products (AoS vs SoA) ===
Dot AoS (Python loop)                N=300000   mean=20.92 ms  stdev=63.9 µs
Dot SoA (NumPy vectorized)           N=300000   mean=989.9 µs  stdev=144.3 µs

=== Bench 2: Strided vs Contiguous ===
Strided sum (stride=8)               N=4000000  mean=667.7 µs  stdev=138.1 µs
Contiguous copy+sum (stride=8)       N=4000000  mean=1.06 ms  stdev=219.6 µs
Chunked sum (chunk=131072)           N=4000000  mean=840.1 µs  stdev=81.0 µs
Contiguous full sum                  N=4000000  mean=757.2 µs  stdev=33.2 µs

=== Bench 3: Matrix Multiply (naive/blocked/NumPy) ===
MatMul NumPy (BLAS)                  N=192      mean=45.6 µs  stdev=0.0 µs

Saved CSV: results_hpc_locality_20250817_141640.csv
Saved plots (PNG).
```

## Notes
- The AoS→SoA change enables vectorized NumPy operations with contiguous memory access, which typically delivers large speedups versus Python loops.
- The strided/contiguous experiment illustrates when copying to improve locality helps and when it hurts because the copy dominates.
- The BLAS matmul gives a sense of the platform’s performance “roofline.”

