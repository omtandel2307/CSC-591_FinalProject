#include "kernels.h"
#include <cuda_runtime.h>

// ============================================================================
// KERNEL 1: NAIVE GEMM
// ============================================================================
// Each thread computes one output element C[row, col]
// No memory optimization - each thread reads from global memory k times

__global__ void naive_gemm_kernel(const float *a, const float *b, float *c, 
                                   int m, int n, int k) {
    // Column index in output matrix (thread x position)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Row index in output matrix (thread y position)
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (row >= m || col >= n) {
        return;
    }

    // Compute dot product: C[row, col] = sum(A[row, kk] * B[kk, col]) for kk in 0..k-1
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += a[row * k + kk] * b[kk * n + col];
    }
    c[row * n + col] = acc;
}


// ============================================================================
// KERNEL 2: TILED GEMM WITH SHARED MEMORY
// ============================================================================
// Thread blocks cooperatively load TILE_SIZE x TILE_SIZE blocks of A and B
// into shared memory, compute partial products, synchronize between tiles

__global__ void tiled_gemm_kernel(const float *a, const float *b, float *c,
                                  int m, int n, int k) {
    // +1 padding on the column dimension eliminates shared memory bank conflicts
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE + 1];

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global output position for this thread
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    // Accumulator for this thread's output element
    float acc = 0.0f;

    // Number of tiles in k dimension
    int tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    // Process each tile of A and B
    for (int tile_idx = 0; tile_idx < tiles; ++tile_idx) {
        // Global column in A (which is in the k dimension)
        int a_col = tile_idx * TILE_SIZE + tx;
        // Load tile of A into shared memory
        // A is (m x k), so access is A[row, a_col]
        if (row < m && a_col < k) {
            shared_a[ty][tx] = a[row * k + a_col];
        } else {
            shared_a[ty][tx] = 0.0f;
        }

        // Global row in B (which is in the k dimension)
        int b_row = tile_idx * TILE_SIZE + ty;
        // Load tile of B into shared memory
        // B is (k x n), so access is B[b_row, col]
        if (b_row < k && col < n) {
            shared_b[ty][tx] = b[b_row * n + col];
        } else {
            shared_b[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their data
        __syncthreads();

        // Compute partial dot product using shared memory
        // shared_a[ty][kk] contains A[row, tile_idx*TILE_SIZE + kk]
        // shared_b[kk][tx] contains B[tile_idx*TILE_SIZE + kk, col]
        for (int kk = 0; kk < TILE_SIZE; ++kk) {
            acc += shared_a[ty][kk] * shared_b[kk][tx];
        }

        // Synchronize before next tile to ensure shared memory is not overwritten
        __syncthreads();
    }

    // Write result to global memory
    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}


// ============================================================================
// KERNEL 3: REGISTER-BLOCKED GEMM
// ============================================================================
// Each thread computes RB_THREAD_TILE_M x RB_THREAD_TILE_N output elements
// using shared memory tiling in the k dimension and register blocking in output

__global__ void register_blocked_gemm_kernel(const float *a, const float *b, float *c,
                                             int m, int n, int k) {
    // +1 padding on the column dimension eliminates shared memory bank conflicts
    __shared__ float shared_a[RB_BLOCK_TILE_M][RB_TILE_K + 1];
    __shared__ float shared_b[RB_TILE_K][RB_BLOCK_TILE_N + 1];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block's top-left corner in output
    int block_row = blockIdx.y * RB_BLOCK_TILE_M;
    int block_col = blockIdx.x * RB_BLOCK_TILE_N;

    // Thread's starting row and column in output (each thread handles 2x2 elements)
    int row0 = block_row + ty * RB_THREAD_TILE_M;
    int row1 = row0 + 1;
    int col0 = block_col + tx * RB_THREAD_TILE_N;
    int col1 = col0 + 1;

    // Accumulators for the four output elements
    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;

    // Number of k-dimension tiles
    int tiles = (k + RB_TILE_K - 1) / RB_TILE_K;

    // Process each k-dimension tile
    for (int tile_idx = 0; tile_idx < tiles; ++tile_idx) {
        int k_base = tile_idx * RB_TILE_K;

        // Load tile of A into shared memory
        // Each thread loads elements for its rows
        // A is (m x k)
        int a_col = k_base + tx;
        if (row0 < m && a_col < k) {
            shared_a[ty * RB_THREAD_TILE_M][tx] = a[row0 * k + a_col];
        } else {
            shared_a[ty * RB_THREAD_TILE_M][tx] = 0.0f;
        }

        if (row1 < m && a_col < k) {
            shared_a[ty * RB_THREAD_TILE_M + 1][tx] = a[row1 * k + a_col];
        } else {
            shared_a[ty * RB_THREAD_TILE_M + 1][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        // Each thread loads elements for its columns
        // B is (k x n)
        int b_row = k_base + ty;
        if (b_row < k && col0 < n) {
            shared_b[ty][tx * RB_THREAD_TILE_N] = b[b_row * n + col0];
        } else {
            shared_b[ty][tx * RB_THREAD_TILE_N] = 0.0f;
        }

        if (b_row < k && col1 < n) {
            shared_b[ty][tx * RB_THREAD_TILE_N + 1] = b[b_row * n + col1];
        } else {
            shared_b[ty][tx * RB_THREAD_TILE_N + 1] = 0.0f;
        }

        // Synchronize to ensure all shared memory loads complete
        __syncthreads();

        // Compute partial dot products using shared memory
        // Each thread computes 4 partial products for its 2x2 output tile
        for (int kk = 0; kk < RB_TILE_K; ++kk) {
            float a0 = shared_a[ty * RB_THREAD_TILE_M][kk];
            float a1 = shared_a[ty * RB_THREAD_TILE_M + 1][kk];
            float b0 = shared_b[kk][tx * RB_THREAD_TILE_N];
            float b1 = shared_b[kk][tx * RB_THREAD_TILE_N + 1];

            acc00 += a0 * b0;  // C[row0, col0]
            acc01 += a0 * b1;  // C[row0, col1]
            acc10 += a1 * b0;  // C[row1, col0]
            acc11 += a1 * b1;  // C[row1, col1]
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write results to global memory
    // C is (m x n)
    if (row0 < m && col0 < n) {
        c[row0 * n + col0] = acc00;
    }
    if (row0 < m && col1 < n) {
        c[row0 * n + col1] = acc01;
    }
    if (row1 < m && col0 < n) {
        c[row1 * n + col0] = acc10;
    }
    if (row1 < m && col1 < n) {
        c[row1 * n + col1] = acc11;
    }
}
