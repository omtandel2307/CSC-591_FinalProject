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

// ============================================================================
// KERNEL 4: HIGH-PERFORMANCE GEMM CONFIGURATION
// ============================================================================
// 128x128 output tile per block, 8x8 register tile per thread
// Arithmetic intensity: 2*8*8 / ((8+8)*4) = 2.0 FLOP/byte  (vs 0.5 for kernel 3)

#define HPC_TILE_K      16
#define HPC_BLOCK_M     128
#define HPC_BLOCK_N     128
#define HPC_THREAD_M    8
#define HPC_THREAD_N    8
#define HPC_THREADS_X   (HPC_BLOCK_N / HPC_THREAD_N)   // 16
#define HPC_THREADS_Y   (HPC_BLOCK_M / HPC_THREAD_M)   // 16

__global__ void hpc_gemm_kernel(const float *a, const float *b, float *c,
                                int m, int n, int k);

// ============================================================================
// KERNEL 5: ULTRA HIGH-PERFORMANCE GEMM CONFIGURATION
// ============================================================================
// Same 128x128 block tile and 8x8 register tile as kernel 4, but TILE_K = 32
// (double kernel 4's 16).  Doubling the K-tile cuts the number of __syncthreads
// calls in half and doubles the FMA-to-overhead ratio per tile pass.
// __ldg() routes all A/B global reads through the read-only (texture) cache,
// improving hit rate for non-square matrix shapes where the same row/column is
// reused across blocks.

#define ULTRA_TILE_K     32
#define ULTRA_BLOCK_M    128
#define ULTRA_BLOCK_N    128
#define ULTRA_THREAD_M   8
#define ULTRA_THREAD_N   8
#define ULTRA_THREADS_X  (ULTRA_BLOCK_N / ULTRA_THREAD_N)   // 16
#define ULTRA_THREADS_Y  (ULTRA_BLOCK_M / ULTRA_THREAD_M)   // 16

__global__ void ultra_gemm_kernel(const float *a, const float *b, float *c,
                                  int m, int n, int k);

// ============================================================================
// KERNEL 6: TURBO GEMM CONFIGURATION
// ============================================================================
// Same 128x128 block tile and 8x8 register tile as Kernel 5, but with three
// targeted optimizations:
//   1. Interleaved B smem layout: stores B[kr][tx*TN+j] at smem[kr][j*TX+tx]
//      so consecutive tx indices access consecutive smem banks (stride 1) instead
//      of stride 8, eliminating the 4-way bank conflicts present in ultra_gemm.
//   2. A smem padding +1 (stride 33): eliminates the 2-way bank conflicts that
//      the +4 padding (stride 36 = 32+4, 8*36%32=0) caused for ultra_gemm.
//   3. Register prefetching: loads smem → registers for kk+1 while computing
//      kk, hiding shared-memory read latency.

#define TURBO_TILE_K    32
#define TURBO_BLOCK_M   128
#define TURBO_BLOCK_N   128
#define TURBO_THREAD_M  8
#define TURBO_THREAD_N  8
#define TURBO_THREADS_X (TURBO_BLOCK_N / TURBO_THREAD_N)   // 16
#define TURBO_THREADS_Y (TURBO_BLOCK_M / TURBO_THREAD_M)   // 16

__global__ void turbo_gemm_kernel(const float *a, const float *b, float *c,
                                  int m, int n, int k);

#endif // GEMM_KERNELS_H
