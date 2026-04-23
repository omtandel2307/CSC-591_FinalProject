#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

#include <cuda_runtime.h>

// Kernel configuration constants
#define NAIVE_BLOCK_X 16
#define NAIVE_BLOCK_Y 16

#define TILE_SIZE 16

#define RB_TILE_K 16
#define RB_THREAD_TILE_M 2
#define RB_THREAD_TILE_N 2
#define RB_BLOCK_THREADS_X 16
#define RB_BLOCK_THREADS_Y 16
#define RB_BLOCK_TILE_M (RB_BLOCK_THREADS_Y * RB_THREAD_TILE_M)
#define RB_BLOCK_TILE_N (RB_BLOCK_THREADS_X * RB_THREAD_TILE_N)

// ============================================================================
// KERNEL DECLARATIONS
// ============================================================================

/**
 * Naive GEMM kernel: each thread computes one output element
 * C = A * B where A is (m x k) and B is (k x n) and C is (m x n)
 * Each thread independently accesses global memory for all k values
 */
__global__ void naive_gemm_kernel(const float *a, const float *b, float *c, 
                                   int m, int n, int k);

/**
 * Tiled GEMM kernel: uses shared memory for data reuse
 * Divides matrices into TILE_SIZE x TILE_SIZE tiles
 * Each thread block loads and computes one tile of output
 */
__global__ void tiled_gemm_kernel(const float *a, const float *b, float *c,
                                  int m, int n, int k);

/**
 * Register-blocked GEMM kernel: each thread computes multiple output elements
 * stored in registers to maximize data reuse and arithmetic intensity
 * Thread block processes RB_BLOCK_TILE_M x RB_BLOCK_TILE_N output tile
 * Each thread computes RB_THREAD_TILE_M x RB_THREAD_TILE_N elements
 */
__global__ void register_blocked_gemm_kernel(const float *a, const float *b, float *c,
                                             int m, int n, int k);

#endif // GEMM_KERNELS_H
