from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
from numba import cuda, njit

from .kernels import KERNEL_SPECS, ensure_float32_contiguous, launch_gemm
from .kernels import RB_THREAD_TILE_M, RB_THREAD_TILE_N, TILE


DEFAULT_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (4096, 256, 256),
    (256, 4096, 256),
]

QUICK_SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (1024, 128, 256),
]


@dataclass
class BenchmarkResult:
    kernel: str
    shape: tuple[int, int, int]
    milliseconds: float
    gflops: float
    max_abs_error: float | None


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    device_name: str
    capability: str | None = None


def make_inputs(m: int, n: int, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = ensure_float32_contiguous(rng.standard_normal((m, k), dtype=np.float32))
    b = ensure_float32_contiguous(rng.standard_normal((k, n), dtype=np.float32))
    return a, b


def gflops(m: int, n: int, k: int, milliseconds: float) -> float:
    ops = 2.0 * m * n * k
    return ops / (milliseconds * 1.0e6)


@njit(cache=True)
def naive_gemm_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, k = a.shape
    _, n = b.shape
    c = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        for col in range(n):
            acc = 0.0
            for kk in range(k):
                acc += a[row, kk] * b[kk, col]
            c[row, col] = acc
    return c


@njit(cache=True)
def tiled_gemm_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, k = a.shape
    _, n = b.shape
    c = np.zeros((m, n), dtype=np.float32)

    for row_tile in range(0, m, TILE):
        row_end = min(row_tile + TILE, m)
        for col_tile in range(0, n, TILE):
            col_end = min(col_tile + TILE, n)
            for k_tile in range(0, k, TILE):
                k_end = min(k_tile + TILE, k)
                for row in range(row_tile, row_end):
                    for col in range(col_tile, col_end):
                        acc = c[row, col]
                        for kk in range(k_tile, k_end):
                            acc += a[row, kk] * b[kk, col]
                        c[row, col] = acc
    return c


@njit(cache=True)
def register_blocked_gemm_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, k = a.shape
    _, n = b.shape
    c = np.zeros((m, n), dtype=np.float32)

    for row in range(0, m, RB_THREAD_TILE_M):
        for col in range(0, n, RB_THREAD_TILE_N):
            acc00 = 0.0
            acc01 = 0.0
            acc10 = 0.0
            acc11 = 0.0

            row0_valid = row < m
            row1_valid = row + 1 < m
            col0_valid = col < n
            col1_valid = col + 1 < n

            for kk in range(k):
                a0 = a[row, kk] if row0_valid else 0.0
                a1 = a[row + 1, kk] if row1_valid else 0.0
                b0 = b[kk, col] if col0_valid else 0.0
                b1 = b[kk, col + 1] if col1_valid else 0.0
                acc00 += a0 * b0
                acc01 += a0 * b1
                acc10 += a1 * b0
                acc11 += a1 * b1

            if row0_valid and col0_valid:
                c[row, col] = acc00
            if row0_valid and col1_valid:
                c[row, col + 1] = acc01
            if row1_valid and col0_valid:
                c[row + 1, col] = acc10
            if row1_valid and col1_valid:
                c[row + 1, col + 1] = acc11

    return c


CPU_KERNELS = (
    ("naive", naive_gemm_cpu),
    ("tiled", tiled_gemm_cpu),
    ("register_blocked", register_blocked_gemm_cpu),
)


def managed_from_numpy(x: np.ndarray):
    managed = cuda.managed_array(x.shape, dtype=np.float32)
    managed[:] = x
    return managed


def validate_numba_runtime() -> None:
    try:
        probe = cuda.managed_array((128, 128), dtype=np.float32)
        probe[:] = 0.0
        cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "Numba CUDA is installed, but this machine cannot create reliable managed-memory "
            "allocations for custom kernels. The current draft is best run in Google Colab or "
            "a local Linux/WSL environment with a full CUDA toolchain."
        ) from exc


def detect_runtime() -> RuntimeConfig:
    if not torch.cuda.is_available():
        return RuntimeConfig(backend="cpu", device_name="CPU")

    try:
        validate_numba_runtime()
    except Exception:
        return RuntimeConfig(
            backend="cpu",
            device_name=f"{torch.cuda.get_device_name(0)} (CUDA unavailable for Numba, using CPU fallback)",
            capability=str(torch.cuda.get_device_capability(0)),
        )

    return RuntimeConfig(
        backend="cuda",
        device_name=torch.cuda.get_device_name(0),
        capability=str(torch.cuda.get_device_capability(0)),
    )


def time_numba_kernel(spec, a_dev, b_dev, c_dev, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        launch_gemm(spec, a_dev, b_dev, c_dev)
    cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        launch_gemm(spec, a_dev, b_dev, c_dev)
    cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def time_torch_matmul(a_host: np.ndarray, b_host: np.ndarray, warmup: int, iters: int) -> float:
    a = torch.from_numpy(a_host).to(device="cuda")
    b = torch.from_numpy(b_host).to(device="cuda")

    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_torch_matmul_cpu(a_host: np.ndarray, b_host: np.ndarray, warmup: int, iters: int) -> float:
    a = torch.from_numpy(a_host)
    b = torch.from_numpy(b_host)

    for _ in range(warmup):
        torch.matmul(a, b)

    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    return (time.perf_counter() - start) * 1000.0 / iters


def verify_kernel(spec, a_host: np.ndarray, b_host: np.ndarray, atol: float, rtol: float) -> float:
    a_dev = managed_from_numpy(a_host)
    b_dev = managed_from_numpy(b_host)
    c_dev = cuda.managed_array((a_host.shape[0], b_host.shape[1]), dtype=np.float32)
    c_dev[:] = 0.0

    launch_gemm(spec, a_dev, b_dev, c_dev)
    cuda.synchronize()

    actual = np.array(c_dev, copy=True)
    reference = torch.matmul(
        torch.from_numpy(a_host).to(device="cuda"),
        torch.from_numpy(b_host).to(device="cuda"),
    ).cpu().numpy()

    if not np.allclose(actual, reference, atol=atol, rtol=rtol):
        diff = np.abs(actual - reference)
        raise AssertionError(
            f"{spec.name} failed correctness check. max_abs_error={diff.max():.6f}, "
            f"mean_abs_error={diff.mean():.6f}"
        )

    return float(np.max(np.abs(actual - reference)))


def verify_cpu_kernel(kernel_name: str, kernel_fn, a_host: np.ndarray, b_host: np.ndarray, atol: float, rtol: float) -> float:
    actual = kernel_fn(a_host, b_host)
    reference = torch.matmul(torch.from_numpy(a_host), torch.from_numpy(b_host)).numpy()

    if not np.allclose(actual, reference, atol=atol, rtol=rtol):
        diff = np.abs(actual - reference)
        raise AssertionError(
            f"{kernel_name} failed correctness check. max_abs_error={diff.max():.6f}, "
            f"mean_abs_error={diff.mean():.6f}"
        )

    return float(np.max(np.abs(actual - reference)))


def time_cpu_kernel(kernel_fn, a_host: np.ndarray, b_host: np.ndarray, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        kernel_fn(a_host, b_host)

    start = time.perf_counter()
    for _ in range(iters):
        kernel_fn(a_host, b_host)
    return (time.perf_counter() - start) * 1000.0 / iters


def benchmark_shape(m: int, n: int, k: int, warmup: int, iters: int, verify: bool) -> list[BenchmarkResult]:
    a_host, b_host = make_inputs(m, n, k, seed=m + n + k)
    results: list[BenchmarkResult] = []

    for spec in KERNEL_SPECS:
        max_abs_error = verify_kernel(spec, a_host, b_host, atol=1e-3, rtol=1e-3) if verify else None
        a_dev = managed_from_numpy(a_host)
        b_dev = managed_from_numpy(b_host)
        c_dev = cuda.managed_array((m, n), dtype=np.float32)
        c_dev[:] = 0.0
        ms = time_numba_kernel(spec, a_dev, b_dev, c_dev, warmup=warmup, iters=iters)
        results.append(BenchmarkResult(spec.name, (m, n, k), ms, gflops(m, n, k, ms), max_abs_error))

    torch_ms = time_torch_matmul(a_host, b_host, warmup=warmup, iters=iters)
    results.append(BenchmarkResult("torch.matmul", (m, n, k), torch_ms, gflops(m, n, k, torch_ms), None))
    return results


def benchmark_shape_cpu(m: int, n: int, k: int, warmup: int, iters: int, verify: bool) -> list[BenchmarkResult]:
    a_host, b_host = make_inputs(m, n, k, seed=m + n + k)
    results: list[BenchmarkResult] = []

    for kernel_name, kernel_fn in CPU_KERNELS:
        max_abs_error = verify_cpu_kernel(kernel_name, kernel_fn, a_host, b_host, atol=1e-3, rtol=1e-3) if verify else None
        ms = time_cpu_kernel(kernel_fn, a_host, b_host, warmup=warmup, iters=iters)
        results.append(BenchmarkResult(kernel_name, (m, n, k), ms, gflops(m, n, k, ms), max_abs_error))

    torch_ms = time_torch_matmul_cpu(a_host, b_host, warmup=warmup, iters=iters)
    results.append(BenchmarkResult("torch.matmul", (m, n, k), torch_ms, gflops(m, n, k, torch_ms), None))
    return results


def print_results(results: list[BenchmarkResult]) -> None:
    print(f"{'kernel':18} {'shape (m,n,k)':18} {'ms':>10} {'GFLOP/s':>12} {'max_abs_error':>15}")
    print("-" * 78)
    for result in results:
        shape = f"{result.shape[0]}x{result.shape[1]}x{result.shape[2]}"
        error = "-" if result.max_abs_error is None else f"{result.max_abs_error:.3e}"
        print(
            f"{result.kernel:18} {shape:18} "
            f"{result.milliseconds:10.3f} {result.gflops:12.2f} {error:>15}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark staged CUDA GEMM kernels.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller benchmark sweep.")
    parser.add_argument("--no-verify", action="store_true", help="Skip numerical correctness checks.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing.")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations per kernel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = detect_runtime()

    shapes = QUICK_SHAPES if args.quick else DEFAULT_SHAPES
    all_results: list[BenchmarkResult] = []

    print(f"Backend: {runtime.backend}")
    print(f"Device: {runtime.device_name}")
    if runtime.capability is not None:
        print(f"CUDA capability: {runtime.capability}")
    print()

    for m, n, k in shapes:
        print(f"Running shape m={m}, n={n}, k={k}")
        if runtime.backend == "cuda":
            results = benchmark_shape(m, n, k, warmup=args.warmup, iters=args.iters, verify=not args.no_verify)
        else:
            results = benchmark_shape_cpu(m, n, k, warmup=args.warmup, iters=args.iters, verify=not args.no_verify)
        print_results(results)
        print()
        all_results.extend(results)

    fastest_custom = max((r for r in all_results if r.kernel != "torch.matmul"), key=lambda x: x.gflops)
    print(
        "Best custom kernel: "
        f"{fastest_custom.kernel} at shape {fastest_custom.shape} "
        f"with {fastest_custom.gflops:.2f} GFLOP/s"
    )


if __name__ == "__main__":
    main()
