# Advanced Computer Architecture Project

## Title
**Performance Analysis and Optimization of a Multithreaded Application Using Intel VTune**

## Overview
This project explores the performance characteristics of a multithreaded application using Intel VTune Profiler. It aims to identify performance bottlenecks and implement optimizations to enhance overall execution efficiency.

The project was developed as part of the curriculum for the Advanced Computer Architecture course.

## Objectives
- Profile a multithreaded application using Intel VTune.
- Analyze performance metrics such as CPU utilization, thread concurrency, memory usage, and hotspots.
- Identify bottlenecks including thread contention, cache misses, and load imbalance.
- Apply appropriate code-level and system-level optimizations.
- Compare performance before and after optimization.

## Tools and Technologies
- **Intel VTune Profiler**: For in-depth performance analysis.
- **C++**: Programming language used to develop the multithreaded application.
- **POSIX Threads (pthreads)**: For implementing multithreading.
- **Linux Environment**: Target platform for building and testing.

## Folder Structure
- ├── src/ # Source code of the multithreaded application
- ├── reports/ # VTune profiling reports (before and after optimization)
- ├── screenshots/ # Images of analysis from VTune
- ├── optimization_notes/ # Documentation on identified issues and applied fixes
- ├── README.md # Project documentation (this file)
- └── Makefile # Build configuration

## Setup and Usage

### Prerequisites
- Intel VTune Profiler (installed and configured)
- GCC or Clang compiler
- Make utility
- Linux OS (Ubuntu recommended)

### Build the Application
```bash
cd src
make
./multithreaded_app

