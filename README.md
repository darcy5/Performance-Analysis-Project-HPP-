# Advanced Computer Architecture Project

## Title
**Performance Analysis and Optimization of a Multithreaded Application Using Intel VTune**

## Overview
This project focuses on analyzing and optimizing the performance of a matrix multiplication application using Intel VTune Profiler. It demonstrates a progressive optimization approach, beginning with a naive implementation and evolving into a highly efficient version using OpenMP, tiling, and SIMD vectorization.

The work was conducted as part of the Advanced Computer Architecture course.

## Objectives
- Profile a matrix multiplication application using Intel VTune.
- Identify performance bottlenecks and inefficiencies.
- Apply a series of optimizations to enhance performance.
- Evaluate and compare results at each stage of optimization.

## Tools and Technologies
- **Intel VTune Profiler**: For detailed performance analysis.
- **C/C++**: Core programming language.
- **POSIX Threads (pthreads)** and **OpenMP**: For multithreading.
- **SIMD Vectorization**: To enhance data-level parallelism.
- **Linux (Ubuntu)**: Development and testing environment.

## Optimization Stages

The project follows a structured optimization pipeline:

1. **Naive Matrix Multiplication**  
   A standard triple-loop implementation with no optimization. Acts as the baseline for performance comparison.

2. **Tiled Matrix Multiplication**  
   Matrix multiplication with loop tiling (blocking) to improve cache locality and reduce cache misses.

3. **Tiled Matrix Multiplication with Pthreads**  
   Parallelization using POSIX threads, distributing tile-based computations across threads.

4. **Tiled Matrix Multiplication with OpenMP**  
   Migrated to OpenMP for simpler thread management and parallel loop control, improving scalability.

5. **Tiled Matrix Multiplication with Three-Level Tiling**  
   Introduced a three-level (L1, L2, L3 cache-aware) tiling strategy to maximize cache reuse and minimize memory traffic.

6. **Tiled Matrix Multiplication with OpenMP + SIMD Vectorization**  
   Combined OpenMP with compiler-level SIMD intrinsics or vectorization pragmas to exploit both thread-level and data-level parallelism for maximum performance.

Each version was profiled with VTune to observe improvements in:
- CPU Utilization
- Memory Access Efficiency
- Thread Load Balance
- Execution Time

## Folder Structure
```
├── src/ # Source code of each version
│ ├── naive/ # Naive matrix multiplication
│ ├── tiled/ # Basic tiling
│ ├── tiled_pthreads/ # Tiling with pthreads
│ ├── tiled_openmp/ # Tiling with OpenMP
│ ├── tiled_3tile/ # Three-level tiled approach
│ └── tiled_simd/ # OpenMP + SIMD optimized
├── reports/ # VTune performance reports
├── screenshots/ # VTune visualizations
├── optimization_notes/ # Notes on strategies and changes
├── README.md # This file
└── Makefile # Build automation
```
## Setup and Usage

### Prerequisites
- Intel VTune Profiler
- GCC or Clang with OpenMP and SIMD support
- Make utility
- Linux OS (Ubuntu recommended)

### Build and Run
```bash
cd src/tiled_simd
make
./matrix_mul

