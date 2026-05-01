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


// ============================================================================
// KERNEL 4: HIGH-PERFORMANCE GEMM
// ============================================================================
// Each block covers a 128x128 output tile.
// Each thread accumulates an 8x8 register tile, giving 16x higher arithmetic
// intensity than kernel 3 (2x2 tiles).  fmaf maps directly to the GPU's FMA
// unit.  __launch_bounds__ tells the register allocator to stay within limits
// that allow 2 blocks per SM.

__global__
__launch_bounds__(HPC_THREADS_X * HPC_THREADS_Y, 2)
void hpc_gemm_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    // Shared memory with +1 column padding to eliminate bank conflicts on loads
    __shared__ float shared_a[HPC_BLOCK_M][HPC_TILE_K + 1];
    __shared__ float shared_b[HPC_TILE_K][HPC_BLOCK_N + 1];

    const int tx  = threadIdx.x;                     // 0..15
    const int ty  = threadIdx.y;                     // 0..15
    const int tid = ty * HPC_THREADS_X + tx;         // 0..255

    const int block_row = blockIdx.y * HPC_BLOCK_M;
    const int block_col = blockIdx.x * HPC_BLOCK_N;

    // 8x8 accumulators live entirely in registers for the full k loop
    float acc[HPC_THREAD_M][HPC_THREAD_N] = {};

    const int num_tiles = (k + HPC_TILE_K - 1) / HPC_TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int k_base = tile * HPC_TILE_K;

        // ── Load A tile: 128 rows × 16 cols = 2048 elements, 8 per thread ──
        // Mapping: lc = tid % HPC_TILE_K  →  consecutive threads hit consecutive
        // k-columns of the same row → coalesced global reads.
        #pragma unroll
        for (int i = 0; i < (HPC_BLOCK_M * HPC_TILE_K) / (HPC_THREADS_X * HPC_THREADS_Y); ++i) {
            const int flat = i * (HPC_THREADS_X * HPC_THREADS_Y) + tid;
            const int lr   = flat / HPC_TILE_K;
            const int lc   = flat % HPC_TILE_K;
            const int gr   = block_row + lr;
            const int gc   = k_base   + lc;
            shared_a[lr][lc] = (gr < m && gc < k) ? a[gr * k + gc] : 0.0f;
        }

        // ── Load B tile: 16 rows × 128 cols = 2048 elements, 8 per thread ──
        // Mapping: lc = tid % HPC_BLOCK_N  →  consecutive threads hit consecutive
        // n-columns of the same row → coalesced global reads.
        #pragma unroll
        for (int i = 0; i < (HPC_TILE_K * HPC_BLOCK_N) / (HPC_THREADS_X * HPC_THREADS_Y); ++i) {
            const int flat = i * (HPC_THREADS_X * HPC_THREADS_Y) + tid;
            const int lr   = flat / HPC_BLOCK_N;
            const int lc   = flat % HPC_BLOCK_N;
            const int gr   = k_base    + lr;
            const int gc   = block_col + lc;
            shared_b[lr][lc] = (gr < k && gc < n) ? b[gr * n + gc] : 0.0f;
        }

        __syncthreads();

        // ── Compute 8x8 outer products across the k tile ──
        // a_reg and b_reg are kept in registers; the compiler sees a fully
        // unrolled 8x8x16 = 1024 FMA body with no shared-memory traffic.
        #pragma unroll
        for (int kk = 0; kk < HPC_TILE_K; ++kk) {
            float a_reg[HPC_THREAD_M];
            float b_reg[HPC_THREAD_N];

            #pragma unroll
            for (int i = 0; i < HPC_THREAD_M; ++i)
                a_reg[i] = shared_a[ty * HPC_THREAD_M + i][kk];

            #pragma unroll
            for (int j = 0; j < HPC_THREAD_N; ++j)
                b_reg[j] = shared_b[kk][tx * HPC_THREAD_N + j];

            #pragma unroll
            for (int i = 0; i < HPC_THREAD_M; ++i)
                #pragma unroll
                for (int j = 0; j < HPC_THREAD_N; ++j)
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
        }

        __syncthreads();
    }

    // ── Write 8x8 tile to global memory ──
    const int out_row = block_row + ty * HPC_THREAD_M;
    const int out_col = block_col + tx * HPC_THREAD_N;

    #pragma unroll
    for (int i = 0; i < HPC_THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < HPC_THREAD_N; ++j) {
            const int r = out_row + i;
            const int c2 = out_col + j;
            if (r < m && c2 < n)
                c[r * n + c2] = acc[i][j];
        }
    }
}


// ============================================================================
// KERNEL 5: ULTRA HIGH-PERFORMANCE GEMM
// ============================================================================
// Extends kernel 4 with a doubled K-tile (32 vs 16).  Each tile pass now
// performs 8*8*32 = 2048 FMAs instead of 1024, halving the number of
// __syncthreads() calls and the fraction of time spent on tile bookkeeping.
//
// __ldg() reads A and B via the read-only (texture/L1) cache.  For tall-skinny
// matrices (large m, small n) the same B rows are reused by many thread blocks;
// the cache captures this reuse and cuts effective global-memory bandwidth.
//
// Shared memory per block:
//   A tile: 128 x 36 floats = 18 432 bytes
//   B tile:  32 x 132 floats = 16 896 bytes
//   Total  ~ 34.5 KB  (fits in the 48 KB default carveout on Turing/Ampere)
//
// __launch_bounds__(256, 1): one block minimum per SM lets the register
// allocator use more registers, keeping the 8x8 accumulators in registers.

__global__
__launch_bounds__(ULTRA_THREADS_X * ULTRA_THREADS_Y, 1)
void ultra_gemm_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    // +4 column padding eliminates bank conflicts on the float4-aligned stores
    __shared__ float shared_a[ULTRA_BLOCK_M][ULTRA_TILE_K + 4];
    __shared__ float shared_b[ULTRA_TILE_K][ULTRA_BLOCK_N + 4];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * ULTRA_THREADS_X + tx;

    const int block_row = blockIdx.y * ULTRA_BLOCK_M;
    const int block_col = blockIdx.x * ULTRA_BLOCK_N;

    float acc[ULTRA_THREAD_M][ULTRA_THREAD_N] = {};

    const int num_tiles = (k + ULTRA_TILE_K - 1) / ULTRA_TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int k_base = tile * ULTRA_TILE_K;

        // ── Load A tile: 128 rows x 32 cols = 4096 elements, 16 per thread ──
        // Consecutive threads stride across k-columns of the same row
        // → coalesced global reads.  __ldg uses the read-only cache.
        #pragma unroll
        for (int i = 0; i < (ULTRA_BLOCK_M * ULTRA_TILE_K) / (ULTRA_THREADS_X * ULTRA_THREADS_Y); ++i) {
            const int flat = i * (ULTRA_THREADS_X * ULTRA_THREADS_Y) + tid;
            const int lr   = flat / ULTRA_TILE_K;
            const int lc   = flat % ULTRA_TILE_K;
            const int gr   = block_row + lr;
            const int gc   = k_base   + lc;
            shared_a[lr][lc] = (gr < m && gc < k) ? __ldg(&a[gr * k + gc]) : 0.0f;
        }

        // ── Load B tile: 32 rows x 128 cols = 4096 elements, 16 per thread ──
        #pragma unroll
        for (int i = 0; i < (ULTRA_TILE_K * ULTRA_BLOCK_N) / (ULTRA_THREADS_X * ULTRA_THREADS_Y); ++i) {
            const int flat = i * (ULTRA_THREADS_X * ULTRA_THREADS_Y) + tid;
            const int lr   = flat / ULTRA_BLOCK_N;
            const int lc   = flat % ULTRA_BLOCK_N;
            const int gr   = k_base    + lr;
            const int gc   = block_col + lc;
            shared_b[lr][lc] = (gr < k && gc < n) ? __ldg(&b[gr * n + gc]) : 0.0f;
        }

        __syncthreads();

        // ── Compute 8x8 outer products across the K=32 tile ──
        // 2048 FMAs per tile with no shared-memory traffic inside the loop.
        #pragma unroll
        for (int kk = 0; kk < ULTRA_TILE_K; ++kk) {
            float a_reg[ULTRA_THREAD_M];
            float b_reg[ULTRA_THREAD_N];

            #pragma unroll
            for (int i = 0; i < ULTRA_THREAD_M; ++i)
                a_reg[i] = shared_a[ty * ULTRA_THREAD_M + i][kk];

            #pragma unroll
            for (int j = 0; j < ULTRA_THREAD_N; ++j)
                b_reg[j] = shared_b[kk][tx * ULTRA_THREAD_N + j];

            #pragma unroll
            for (int i = 0; i < ULTRA_THREAD_M; ++i)
                #pragma unroll
                for (int j = 0; j < ULTRA_THREAD_N; ++j)
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
        }

        __syncthreads();
    }

    // ── Write 8x8 tile to global memory ──
    const int out_row = block_row + ty * ULTRA_THREAD_M;
    const int out_col = block_col + tx * ULTRA_THREAD_N;

    #pragma unroll
    for (int i = 0; i < ULTRA_THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < ULTRA_THREAD_N; ++j) {
            const int r  = out_row + i;
            const int c2 = out_col + j;
            if (r < m && c2 < n)
                c[r * n + c2] = acc[i][j];
        }
    }
}


// ============================================================================
// KERNEL 6: TURBO GEMM
// ============================================================================
// Fixes the two bank-conflict sources in ultra_gemm and adds register
// prefetching to overlap smem reads with FMA execution.
//
// Bank conflict analysis for B in ultra_gemm:
//   shared_b[kk][tx*8+j], BN+4=132 columns.
//   bank = (kk*132 + tx*8 + j) % 32.  kk*132%32 = kk*4.
//   For 16 tx values (stride 8): tx=0,1,...,15 hit banks
//   {kk*4+0, kk*4+8, kk*4+16, kk*4+24, kk*4+0, ...} → 4-way conflict.
//
// Fix: store B as shared_b[kr][j*TX+tx] where TX=16, TN=8.
//   shared_b[kr][j*TX+tx] = B[k_base+kr][block_col + tx*TN + j]
//   Read: b_reg[j] = shared_b[kk][j*TX+tx]  (consecutive tx → consecutive banks, no conflict)
//   Load: b_col = (lc % TX) * TN + (lc / TX)  (non-consecutive global reads, but
//         smem reads happen TILE_K=32 times per tile vs global load once → tradeoff favors smem)
//
// Bank conflict analysis for A in ultra_gemm:
//   shared_a[128][TILE_K+4=36], stride 36.  8*36%32 = 288%32 = 0 → rows 0 and 8 share banks.
//   Fix: padding +1 gives stride 33.  8*33%32 = 264%32 = 8 ≠ 0 → all rows access distinct banks.
//
// Register prefetch: load kk+1 slice while computing kk, hiding smem latency.

__global__
__launch_bounds__(TURBO_THREADS_X * TURBO_THREADS_Y, 1)
void turbo_gemm_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    // +1 padding: stride 33, eliminates 2-way bank conflicts (8*33%32=8)
    __shared__ float shared_a[TURBO_BLOCK_M][TURBO_TILE_K + 1];
    // Interleaved B layout: shared_b[kr][j*TX+tx] = B[kr][tx*TN+j]
    __shared__ float shared_b[TURBO_TILE_K][TURBO_BLOCK_N];

    const int tx  = threadIdx.x;   // 0..15
    const int ty  = threadIdx.y;   // 0..15
    const int tid = ty * TURBO_THREADS_X + tx;

    const int block_row = blockIdx.y * TURBO_BLOCK_M;
    const int block_col = blockIdx.x * TURBO_BLOCK_N;

    float acc[TURBO_THREAD_M][TURBO_THREAD_N] = {};

    const int NTHREADS   = TURBO_THREADS_X * TURBO_THREADS_Y;  // 256
    const int num_tiles  = (k + TURBO_TILE_K - 1) / TURBO_TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int k_base = tile * TURBO_TILE_K;

        // ── Load A tile: 128×32 = 4096 elements, 16 per thread, coalesced ──
        #pragma unroll
        for (int i = 0; i < (TURBO_BLOCK_M * TURBO_TILE_K) / NTHREADS; ++i) {
            const int flat = i * NTHREADS + tid;
            const int lr   = flat / TURBO_TILE_K;
            const int lc   = flat % TURBO_TILE_K;
            const int gr   = block_row + lr;
            const int gc   = k_base   + lc;
            shared_a[lr][lc] = (gr < m && gc < k) ? __ldg(&a[gr * k + gc]) : 0.0f;
        }

        // ── Load B tile (interleaved): 32×128 = 4096 elements, 16 per thread ──
        // Interleaved column mapping: smem column lc → B column (lc%TX)*TN + lc/TX
        // Consecutive lc → consecutive smem banks; global loads stride by TN=8
        // (acceptable: global loads happen once per tile, smem reads happen TILE_K=32 times)
        #pragma unroll
        for (int i = 0; i < (TURBO_TILE_K * TURBO_BLOCK_N) / NTHREADS; ++i) {
            const int flat  = i * NTHREADS + tid;
            const int lr    = flat / TURBO_BLOCK_N;
            const int lc    = flat % TURBO_BLOCK_N;
            const int b_col = (lc % TURBO_THREADS_X) * TURBO_THREAD_N + (lc / TURBO_THREADS_X);
            const int gr    = k_base    + lr;
            const int gc    = block_col + b_col;
            shared_b[lr][lc] = (gr < k && gc < n) ? __ldg(&b[gr * n + gc]) : 0.0f;
        }

        __syncthreads();

        // ── Prefetch kk=0 slice into registers ──
        float a_cur[TURBO_THREAD_M], b_cur[TURBO_THREAD_N];
        #pragma unroll
        for (int i = 0; i < TURBO_THREAD_M; ++i)
            a_cur[i] = shared_a[ty * TURBO_THREAD_M + i][0];
        #pragma unroll
        for (int j = 0; j < TURBO_THREAD_N; ++j)
            b_cur[j] = shared_b[0][j * TURBO_THREADS_X + tx];

        // ── Compute 8x8 outer products with register prefetch ──
        // Prefetch kk+1 while computing kk to hide smem read latency
        #pragma unroll
        for (int kk = 0; kk < TURBO_TILE_K - 1; ++kk) {
            float a_next[TURBO_THREAD_M], b_next[TURBO_THREAD_N];

            #pragma unroll
            for (int i = 0; i < TURBO_THREAD_M; ++i)
                a_next[i] = shared_a[ty * TURBO_THREAD_M + i][kk + 1];
            #pragma unroll
            for (int j = 0; j < TURBO_THREAD_N; ++j)
                b_next[j] = shared_b[kk + 1][j * TURBO_THREADS_X + tx];

            #pragma unroll
            for (int i = 0; i < TURBO_THREAD_M; ++i)
                #pragma unroll
                for (int j = 0; j < TURBO_THREAD_N; ++j)
                    acc[i][j] = fmaf(a_cur[i], b_cur[j], acc[i][j]);

            #pragma unroll
            for (int i = 0; i < TURBO_THREAD_M; ++i) a_cur[i] = a_next[i];
            #pragma unroll
            for (int j = 0; j < TURBO_THREAD_N; ++j) b_cur[j] = b_next[j];
        }
        // Last kk = TURBO_TILE_K - 1
        #pragma unroll
        for (int i = 0; i < TURBO_THREAD_M; ++i)
            #pragma unroll
            for (int j = 0; j < TURBO_THREAD_N; ++j)
                acc[i][j] = fmaf(a_cur[i], b_cur[j], acc[i][j]);

        __syncthreads();
    }

    // ── Write 8x8 tile to global memory ──
    const int out_row = block_row + ty * TURBO_THREAD_M;
    const int out_col = block_col + tx * TURBO_THREAD_N;

    #pragma unroll
    for (int i = 0; i < TURBO_THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < TURBO_THREAD_N; ++j) {
            const int r  = out_row + i;
            const int c2 = out_col + j;
            if (r < m && c2 < n)
                c[r * n + c2] = acc[i][j];
        }
    }
}
