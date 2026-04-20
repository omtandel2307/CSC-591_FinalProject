from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import cuda, float32


NAIVE_BLOCK = (16, 16)
TILE = 16
RB_TILE_K = 16
RB_THREAD_TILE_M = 2
RB_THREAD_TILE_N = 2
RB_BLOCK_THREADS = (16, 16)
RB_BLOCK_TILE_M = RB_BLOCK_THREADS[1] * RB_THREAD_TILE_M
RB_BLOCK_TILE_N = RB_BLOCK_THREADS[0] * RB_THREAD_TILE_N


@cuda.jit
def naive_gemm_kernel(a, b, c, m, n, k):
    col, row = cuda.grid(2)
    if row >= m or col >= n:
        return

    acc = 0.0
    for kk in range(k):
        acc += a[row, kk] * b[kk, col]
    c[row, col] = acc


@cuda.jit
def tiled_gemm_kernel(a, b, c, m, n, k):
    shared_a = cuda.shared.array((TILE, TILE), dtype=float32)
    shared_b = cuda.shared.array((TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + tx
    row = cuda.blockIdx.y * TILE + ty

    acc = 0.0
    tiles = (k + TILE - 1) // TILE

    for tile_idx in range(tiles):
        a_col = tile_idx * TILE + tx
        b_row = tile_idx * TILE + ty

        if row < m and a_col < k:
            shared_a[ty, tx] = a[row, a_col]
        else:
            shared_a[ty, tx] = 0.0

        if b_row < k and col < n:
            shared_b[ty, tx] = b[b_row, col]
        else:
            shared_b[ty, tx] = 0.0

        cuda.syncthreads()

        for kk in range(TILE):
            acc += shared_a[ty, kk] * shared_b[kk, tx]

        cuda.syncthreads()

    if row < m and col < n:
        c[row, col] = acc


@cuda.jit
def register_blocked_gemm_kernel(a, b, c, m, n, k):
    shared_a = cuda.shared.array((RB_BLOCK_TILE_M, RB_TILE_K), dtype=float32)
    shared_b = cuda.shared.array((RB_TILE_K, RB_BLOCK_TILE_N + 1), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    block_row = cuda.blockIdx.y * RB_BLOCK_TILE_M
    block_col = cuda.blockIdx.x * RB_BLOCK_TILE_N

    row0 = block_row + ty * RB_THREAD_TILE_M
    row1 = row0 + 1
    col0 = block_col + tx * RB_THREAD_TILE_N
    col1 = col0 + 1

    acc00 = 0.0
    acc01 = 0.0
    acc10 = 0.0
    acc11 = 0.0

    tiles = (k + RB_TILE_K - 1) // RB_TILE_K

    for tile_idx in range(tiles):
        k_base = tile_idx * RB_TILE_K

        a_col = k_base + tx
        if row0 < m and a_col < k:
            shared_a[ty * RB_THREAD_TILE_M, tx] = a[row0, a_col]
        else:
            shared_a[ty * RB_THREAD_TILE_M, tx] = 0.0
        if row1 < m and a_col < k:
            shared_a[ty * RB_THREAD_TILE_M + 1, tx] = a[row1, a_col]
        else:
            shared_a[ty * RB_THREAD_TILE_M + 1, tx] = 0.0

        b_row = k_base + ty
        if b_row < k and col0 < n:
            shared_b[ty, tx * RB_THREAD_TILE_N] = b[b_row, col0]
        else:
            shared_b[ty, tx * RB_THREAD_TILE_N] = 0.0
        if b_row < k and col1 < n:
            shared_b[ty, tx * RB_THREAD_TILE_N + 1] = b[b_row, col1]
        else:
            shared_b[ty, tx * RB_THREAD_TILE_N + 1] = 0.0

        cuda.syncthreads()

        for kk in range(RB_TILE_K):
            a0 = shared_a[ty * RB_THREAD_TILE_M, kk]
            a1 = shared_a[ty * RB_THREAD_TILE_M + 1, kk]
            b0 = shared_b[kk, tx * RB_THREAD_TILE_N]
            b1 = shared_b[kk, tx * RB_THREAD_TILE_N + 1]
            acc00 += a0 * b0
            acc01 += a0 * b1
            acc10 += a1 * b0
            acc11 += a1 * b1

        cuda.syncthreads()

    if row0 < m and col0 < n:
        c[row0, col0] = acc00
    if row0 < m and col1 < n:
        c[row0, col1] = acc01
    if row1 < m and col0 < n:
        c[row1, col0] = acc10
    if row1 < m and col1 < n:
        c[row1, col1] = acc11


@cuda.jit
def register_blocked_unrolled_gemm_kernel(a, b, c, m, n, k):
    shared_a = cuda.shared.array((RB_BLOCK_TILE_M, RB_TILE_K), dtype=float32)
    shared_b = cuda.shared.array((RB_TILE_K, RB_BLOCK_TILE_N + 1), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    block_row = cuda.blockIdx.y * RB_BLOCK_TILE_M
    block_col = cuda.blockIdx.x * RB_BLOCK_TILE_N

    row0 = block_row + ty * RB_THREAD_TILE_M
    row1 = row0 + 1
    col0 = block_col + tx * RB_THREAD_TILE_N
    col1 = col0 + 1

    acc00 = 0.0
    acc01 = 0.0
    acc10 = 0.0
    acc11 = 0.0

    tiles = (k + RB_TILE_K - 1) // RB_TILE_K

    for tile_idx in range(tiles):
        k_base = tile_idx * RB_TILE_K

        a_col = k_base + tx
        if row0 < m and a_col < k:
            shared_a[ty * RB_THREAD_TILE_M, tx] = a[row0, a_col]
        else:
            shared_a[ty * RB_THREAD_TILE_M, tx] = 0.0
        if row1 < m and a_col < k:
            shared_a[ty * RB_THREAD_TILE_M + 1, tx] = a[row1, a_col]
        else:
            shared_a[ty * RB_THREAD_TILE_M + 1, tx] = 0.0

        b_row = k_base + ty
        if b_row < k and col0 < n:
            shared_b[ty, tx * RB_THREAD_TILE_N] = b[b_row, col0]
        else:
            shared_b[ty, tx * RB_THREAD_TILE_N] = 0.0
        if b_row < k and col1 < n:
            shared_b[ty, tx * RB_THREAD_TILE_N + 1] = b[b_row, col1]
        else:
            shared_b[ty, tx * RB_THREAD_TILE_N + 1] = 0.0

        cuda.syncthreads()

        for kk in range(0, RB_TILE_K, 4):
            a00 = shared_a[ty * RB_THREAD_TILE_M, kk]
            a01 = shared_a[ty * RB_THREAD_TILE_M, kk + 1]
            a02 = shared_a[ty * RB_THREAD_TILE_M, kk + 2]
            a03 = shared_a[ty * RB_THREAD_TILE_M, kk + 3]
            a10 = shared_a[ty * RB_THREAD_TILE_M + 1, kk]
            a11 = shared_a[ty * RB_THREAD_TILE_M + 1, kk + 1]
            a12 = shared_a[ty * RB_THREAD_TILE_M + 1, kk + 2]
            a13 = shared_a[ty * RB_THREAD_TILE_M + 1, kk + 3]

            b00 = shared_b[kk, tx * RB_THREAD_TILE_N]
            b01 = shared_b[kk, tx * RB_THREAD_TILE_N + 1]
            b10 = shared_b[kk + 1, tx * RB_THREAD_TILE_N]
            b11 = shared_b[kk + 1, tx * RB_THREAD_TILE_N + 1]
            b20 = shared_b[kk + 2, tx * RB_THREAD_TILE_N]
            b21 = shared_b[kk + 2, tx * RB_THREAD_TILE_N + 1]
            b30 = shared_b[kk + 3, tx * RB_THREAD_TILE_N]
            b31 = shared_b[kk + 3, tx * RB_THREAD_TILE_N + 1]

            acc00 += a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30
            acc01 += a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31
            acc10 += a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30
            acc11 += a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31

        cuda.syncthreads()

    if row0 < m and col0 < n:
        c[row0, col0] = acc00
    if row0 < m and col1 < n:
        c[row0, col1] = acc01
    if row1 < m and col0 < n:
        c[row1, col0] = acc10
    if row1 < m and col1 < n:
        c[row1, col1] = acc11


@dataclass(frozen=True)
class KernelSpec:
    name: str
    kernel: object
    block: tuple[int, int]
    tile_rows: int
    tile_cols: int

    def grid(self, m: int, n: int) -> tuple[int, int]:
        return ((n + self.tile_cols - 1) // self.tile_cols, (m + self.tile_rows - 1) // self.tile_rows)


KERNEL_SPECS = (
    KernelSpec("naive", naive_gemm_kernel, NAIVE_BLOCK, NAIVE_BLOCK[1], NAIVE_BLOCK[0]),
    KernelSpec("tiled", tiled_gemm_kernel, (TILE, TILE), TILE, TILE),
    KernelSpec("register_blocked", register_blocked_gemm_kernel, RB_BLOCK_THREADS, RB_BLOCK_TILE_M, RB_BLOCK_TILE_N),
    KernelSpec(
        "register_blocked_unrolled",
        register_blocked_unrolled_gemm_kernel,
        RB_BLOCK_THREADS,
        RB_BLOCK_TILE_M,
        RB_BLOCK_TILE_N,
    ),
)


def ensure_float32_contiguous(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


def launch_gemm(spec: KernelSpec, a_dev, b_dev, c_dev) -> None:
    m, k = a_dev.shape
    kb, n = b_dev.shape
    if k != kb:
        raise ValueError(f"Incompatible shapes: {a_dev.shape} x {b_dev.shape}")
    grid = spec.grid(m, n)
    spec.kernel[grid, spec.block](a_dev, b_dev, c_dev, m, n, k)
