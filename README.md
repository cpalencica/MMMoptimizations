# High Performance Matrix Multiplication
Authors: Cristian Palencia and Yiran Yin

This project implements and analyzes various matrix-matrix multiplication (MMM) strategies on both CPU and GPU. The goal is to explore performance improvements through different optimization techniques including loop transformations, blocking, OpenMP parallelization, and CUDA acceleration using both global and shared memory.

---

## Files Overview

- `mmm_blocking.c` – CPU implementation of ijk MMM with optional loop blocking and configurable loop order permutations.
- `mmm_openmp.c` – CPU version with OpenMP for multithreaded parallelism.
- `cuda_MMM.cu` – CUDA implementation featuring both global memory and shared memory kernels for GPU acceleration.
- `dense_MMM_SUMMA.c` – Distributed-memory dense matrix multiplication using MPI and OpenMP based on the SUMMA algorithm.

---

## Matrix Multiplication Implementations

### 1. Blocked MMM (`test_mmm_block.c`)

This file includes:
- A basic ijk implementation of matrix multiplication.
- A cache-aware blocked implementation using loop tiling.
- Optional testing of different loop order permutations for performance comparison.

**Highlights:**
- Loop blocking improves spatial and temporal locality.
- Configurable loop order helps analyze performance impact of access patterns.
- Accepts a block size and loop order as arguments.

---

### 2. OpenMP MMM (`test_mmm_omp.c`)

Adds multithreading to the blocked CPU version using OpenMP directives.

**Highlights:**
- Parallelizes the outermost loop with OpenMP.
- Reduces computation time on multicore systems.
- Number of threads is configurable via the OMP_NUM_THREADS environment variable.

---

### 3. CUDA MMM (`cuda_MMM.cu`)

A GPU-accelerated version of matrix multiplication written in CUDA. It contains:

- A global memory kernel: each thread computes a portion of the matrix directly using data from global memory.
- A shared memory kernel: uses tiling and shared memory for faster access and higher throughput.

**Key Features:**
- Dynamically sets matrix size via grid and block configuration.
- Measures GPU time using CUDA events.
- Measures CPU time using clock_gettime.
- Compares GPU output to a blocked CPU version for correctness.

---

### 4. Dense MMM with SUMMA (`dense_MMM_SUMMA.c`)

This implementation performs dense matrix-matrix multiplication on distributed-memory systems using MPI combined with OpenMP for hybrid parallelism. It employs the **Scalable Universal Matrix Multiplication Algorithm (SUMMA)** to efficiently scale matrix multiplication across multiple processes.

#### What is Dense MMM?

Dense MMM involves multiplying two fully populated (dense) matrices without exploiting sparsity. It is computationally intensive and benefits greatly from parallelism on both shared-memory and distributed-memory architectures.

#### SUMMA Algorithm Overview

- Processes are arranged in a logical 2D grid (Pr × Pc).
- Each process owns a block (submatrix) of matrices A, B, and C.
- Computation proceeds in steps, where at step k:
  - The process holding the k-th column block of A broadcasts it to all processes in its row.
  - The process holding the k-th row block of B broadcasts it to all processes in its column.
  - Each process multiplies the received blocks locally and accumulates into its block of C.
  
This approach:
- Limits communication to row and column broadcasts, avoiding expensive all-to-all communication.
- Balances workload evenly by partitioning matrices into equal-sized blocks.
- Overlaps communication and computation to improve scalability.
- Uses OpenMP to parallelize local block multiplications within each process for maximal efficiency.

---

## How to Compile and Run

### CPU Versions

Compile using gcc and link the math and pthread libraries:
- `test_mmm_block.c`
- `test_mmm_omp.c`

Each can be compiled with standard gcc flags. OpenMP requires the appropriate flag.

### CUDA Version

Compile using nvcc:
- `cuda_MMM.cu`

Then run the executable to perform GPU-based matrix multiplication and compare results.

### Dense MMM with MPI and OpenMP

Compile using mpicc with optimization flags for MPI and OpenMP.

Run with mpirun specifying the total number of processes and processor grid dimensions.

---

## Summary

| Version            | Parallel? | Optimized? | Uses GPU? | Uses MPI? | Expected Speedup           |
|--------------------|-----------|------------|-----------|-----------|---------------------------|
| ijk (CPU)          | No        | No         | No        | No        | Baseline                  |
| Blocked            | No        | Yes        | No        | No        | Moderate                  |
| OpenMP             | Yes       | Yes        | No        | No        | High (multi-core)         |
| CUDA (global)      | Yes       | Somewhat   | Yes       | No        | High                      |
| CUDA (shared)      | Yes       | Yes        | Yes       | No        | Very High                 |
| Dense MMM (SUMMA)  | Yes       | Yes        | No        | Yes       | Very High (distributed)   |

---
