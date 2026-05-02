# CUDA GEMM Optimization

This repository contains the CUDA C++ implementation and Google Colab workflow for a CSC 591 GEMM optimization study.

The project benchmarks a staged sequence of GEMM kernels:

1. Naive global-memory GEMM
2. Shared-memory tiled GEMM
3. Register-blocked GEMM
4. HPC GEMM with larger block and thread tiles
5. Ultra GEMM with larger K-tiles and read-only cached loads
6. Turbo GEMM with shared-memory layout tuning and register prefetching

The kernels are compiled with CMake into a shared library and called from Python in the notebook using `ctypes`. PyTorch `matmul` is used as the correctness and performance reference baseline.

## Files

- `CSC591_GEMM_Colab.ipynb`: Colab notebook that clones this repo, builds the CUDA kernels, runs correctness checks, benchmarks performance, plots results, and saves CSV/JSON reports.
- `src/cuda_kernels/kernels.cu`: CUDA kernel implementations.
- `src/cuda_kernels/kernels.h`: Kernel declarations and configuration constants.
- `src/cuda_kernels/kernels_utils.h`: CUDA launch and memory helper utilities.
- `src/cuda_kernels/gemm_c_interface.cu`: C-style wrapper functions used by Python `ctypes`.
- `src/cuda_kernels/gemm_c_interface.h`: C interface declarations.
- `src/cuda_kernels/CMakeLists.txt`: CUDA build configuration.

## How To Run

1. Open `CSC591_GEMM_Colab.ipynb` in Google Colab.
2. Enable a GPU runtime with `Runtime -> Change runtime type -> GPU`.
3. Run all cells.

The notebook installs required Python packages, clones this repository, builds the CUDA shared library, validates the kernels, and benchmarks them against PyTorch.

## Notes

- Inputs are `float32`.
- The custom CUDA wrapper timing includes device allocation and host/device transfers.
- The PyTorch timing is measured after tensors are already placed on the GPU, so the reported percentages are practical end-to-end reference comparisons rather than pure kernel-only efficiency measurements.
- The CUDA kernels guard boundary conditions for matrix dimensions that are not exact multiples of the tile size.
