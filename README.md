# GEMM Optimization

This project implements the staged GEMM plan from the proposal:

1. Naive global-memory GEMM
2. Shared-memory tiled GEMM
3. Register-blocked GEMM

The kernels are written with Numba CUDA so they can run without a local `nvcc` toolchain when CUDA is available. PyTorch is used as the correctness and performance baseline through `torch.matmul`.

## Layout

- `src/gemm_optimization/kernels.py`: CUDA kernels and launch helpers
- `src/gemm_optimization/benchmark.py`: correctness checks and benchmark harness

## Quick start

```powershell
python -m gemm_optimization.benchmark --quick
```

Run a broader comparison:

```powershell
python -m gemm_optimization.benchmark
```

On machines where Numba CUDA cannot initialize reliably, the benchmark automatically falls back to CPU implementations of the same staged kernels so the project still runs end to end.

## Notes

- Inputs are `float32`.
- The tiled and register-blocked kernels assume dimensions are not necessarily multiples of the tile size and guard bounds accordingly.
- A double-buffered kernel is left as a natural next step after validating these three stages.
