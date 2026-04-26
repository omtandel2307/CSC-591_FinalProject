from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .attention import (
    make_attention_inputs,
    materialized_attention_cpu,
    streaming_attention_cpu,
    torch_attention_reference,
)


DEFAULT_SHAPES = [
    (128, 64, 64),
    (256, 64, 64),
    (512, 64, 64),
]

QUICK_SHAPES = [
    (64, 64, 64),
    (128, 64, 64),
    (256, 64, 64),
]


@dataclass
class AttentionBenchmarkResult:
    algorithm: str
    shape: tuple[int, int, int]
    milliseconds: float
    gflops: float
    score_workspace_bytes: int | None
    max_abs_error: float


def attention_gflops(seq_len: int, head_dim: int, value_dim: int, milliseconds: float) -> float:
    qk_ops = 2.0 * seq_len * seq_len * head_dim
    pv_ops = 2.0 * seq_len * seq_len * value_dim
    return (qk_ops + pv_ops) / (milliseconds * 1.0e6)


def materialized_score_workspace_bytes(seq_len: int) -> int:
    return seq_len * seq_len * np.dtype(np.float32).itemsize


def streaming_score_workspace_bytes(block_size: int) -> int:
    return block_size * np.dtype(np.float32).itemsize


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "N/A"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.1f} KB"
    return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def time_kernel(fn: Callable, *args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args)

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - start) * 1000.0 / iters


def max_abs_error(actual: np.ndarray, reference: np.ndarray, name: str, atol: float, rtol: float) -> float:
    diff = np.abs(actual - reference)
    error = float(np.max(diff))

    if not np.allclose(actual, reference, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} failed correctness check. max_abs_error={error:.6f}, "
            f"mean_abs_error={diff.mean():.6f}"
        )

    return error


def verify_kernel(name: str, fn: Callable, q: np.ndarray, k: np.ndarray, v: np.ndarray, reference: np.ndarray) -> float:
    actual = fn(q, k, v)
    return max_abs_error(actual, reference, name, atol=1e-4, rtol=1e-4)


def benchmark_shape(
    seq_len: int,
    head_dim: int,
    value_dim: int,
    warmup: int,
    iters: int,
    verify: bool,
    block_size: int,
) -> list[AttentionBenchmarkResult]:
    q, k, v = make_attention_inputs(seq_len, head_dim, value_dim, seed=seq_len + head_dim + value_dim)
    reference = torch_attention_reference(q, k, v)
    results: list[AttentionBenchmarkResult] = []

    kernels = (
        (
            "materialized_attention",
            materialized_attention_cpu,
            materialized_score_workspace_bytes(seq_len),
        ),
        (
            f"streaming_attention_bs{block_size}",
            lambda q_, k_, v_: streaming_attention_cpu(q_, k_, v_, block_size=block_size),
            streaming_score_workspace_bytes(block_size),
        ),
    )

    for name, fn, score_workspace_bytes in kernels:
        error = verify_kernel(name, fn, q, k, v, reference) if verify else 0.0
        ms = time_kernel(fn, q, k, v, warmup=warmup, iters=iters)
        results.append(
            AttentionBenchmarkResult(
                name,
                (seq_len, head_dim, value_dim),
                ms,
                attention_gflops(seq_len, head_dim, value_dim, ms),
                score_workspace_bytes,
                error,
            )
        )

    torch_ms = time_kernel(torch_attention_reference, q, k, v, warmup=warmup, iters=iters)
    results.append(
        AttentionBenchmarkResult(
            "torch_eager_reference",
            (seq_len, head_dim, value_dim),
            torch_ms,
            attention_gflops(seq_len, head_dim, value_dim, torch_ms),
            None,
            0.0,
        )
    )

    return results


def print_results(results: list[AttentionBenchmarkResult]) -> None:
    print(
        f"{'algorithm':28} {'shape (n,d,dv)':18} {'ms':>10} "
        f"{'GFLOP/s':>10} {'score_workspace':>17} {'max_abs_error':>15}"
    )
    print("-" * 105)
    for result in results:
        shape = f"{result.shape[0]}x{result.shape[1]}x{result.shape[2]}"
        print(
            f"{result.algorithm:28} {shape:18} "
            f"{result.milliseconds:10.3f} {result.gflops:10.2f} "
            f"{format_bytes(result.score_workspace_bytes):>17} {result.max_abs_error:15.3e}"
        )


def print_takeaway(results: list[AttentionBenchmarkResult]) -> None:
    custom_results = [result for result in results if result.score_workspace_bytes is not None]
    fastest = min(custom_results, key=lambda result: result.milliseconds)
    smallest_workspace = min(custom_results, key=lambda result: result.score_workspace_bytes or 0)
    materialized = next(result for result in custom_results if result.algorithm == "materialized_attention")
    streaming = next(result for result in custom_results if result.algorithm.startswith("streaming_attention"))
    reduction = materialized.score_workspace_bytes / streaming.score_workspace_bytes

    print(
        f"Takeaway: fastest custom = {fastest.algorithm}; "
        f"lowest explicit score storage = {smallest_workspace.algorithm}; "
        f"streaming uses {reduction:.0f}x less score workspace than materialized."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compact attention implementations.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller benchmark sweep.")
    parser.add_argument("--no-verify", action="store_true", help="Skip numerical correctness checks.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations before timing.")
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations per kernel.")
    parser.add_argument("--block-size", type=int, default=64, help="Streaming attention block size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = QUICK_SHAPES if args.quick else DEFAULT_SHAPES
    all_results: list[AttentionBenchmarkResult] = []

    print("Attention benchmark")
    print(f"Streaming block size: {args.block_size}")
    print("score_workspace counts only the explicit attention-score storage used by this implementation.")
    print("PyTorch is shown as a timing/reference baseline; its internal memory is not measured here.")
    print()

    for seq_len, head_dim, value_dim in shapes:
        print(f"Running shape seq_len={seq_len}, head_dim={head_dim}, value_dim={value_dim}")
        results = benchmark_shape(
            seq_len,
            head_dim,
            value_dim,
            warmup=args.warmup,
            iters=args.iters,
            verify=not args.no_verify,
            block_size=args.block_size,
        )
        print_results(results)
        print_takeaway(results)
        print()
        all_results.extend(results)


if __name__ == "__main__":
    main()
