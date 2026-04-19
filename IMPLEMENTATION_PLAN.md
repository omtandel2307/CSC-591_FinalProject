# Implementation Plan

This plan maps the proposal into an implementation sequence we can execute in this repo.

## Phase 1: Baseline kernel

- Implement a naive GEMM kernel where each thread computes one output element.
- Validate numerically against `torch.matmul`.
- Measure latency and convert to GFLOP/s.

## Phase 2: Shared-memory tiling

- Introduce thread-block tiles for `A` and `B`.
- Use `__syncthreads()`/block synchronization around each tile load and compute phase.
- Sweep tile sizes and compare arithmetic intensity and throughput to the naive kernel.

## Phase 3: Register blocking

- Expand each thread from one output element to a small output fragment.
- Keep partial sums in registers.
- Compare throughput gains and watch for occupancy regression from register pressure.

## Phase 4: Benchmarking

- Benchmark square and tall-skinny shapes.
- Compare each custom kernel against `torch.matmul`.
- Record max absolute error for correctness and GFLOP/s for speed.

## Phase 5: Profiling

- Run Nsight Compute or Colab profiling tools on each stage.
- Collect occupancy, shared-memory efficiency, memory throughput, and cache behavior.
- Use the results to explain remaining gaps to cuBLAS.

## Current repo status

- Project scaffold created
- Kernel stages drafted
- Benchmark harness added
- Local runtime blocked by this machine's custom-kernel environment

## Best next execution target

The proposal already mentions Google Colab with a T4 GPU. That is the most reliable next place to compile and benchmark custom CUDA kernels if we want full end-to-end validation.
