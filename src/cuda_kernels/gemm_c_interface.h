#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>

/**
 * C-style wrapper functions for calling CUDA kernels from Python
 * All functions take raw pointers and matrix dimensions
 * Memory management (malloc/free) is handled by caller
 */

// ============================================================================
// WRAPPER FUNCTIONS - GEMM COMPUTATION
// ============================================================================

/**
 * Compute C = A * B using naive GEMM kernel
 * 
 * @param h_a     Input matrix A on host (row-major, m x k)
 * @param h_b     Input matrix B on host (row-major, k x n)
 * @param h_c     Output matrix C on host (row-major, m x n)
 * @param m       Number of rows in A and C
 * @param n       Number of columns in B and C
 * @param k       Number of columns in A and rows in B
 * @return        0 on success, non-zero on error
 */
int naive_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k);

/**
 * Compute C = A * B using tiled GEMM kernel
 * 
 * @param h_a     Input matrix A on host (row-major, m x k)
 * @param h_b     Input matrix B on host (row-major, k x n)
 * @param h_c     Output matrix C on host (row-major, m x n)
 * @param m       Number of rows in A and C
 * @param n       Number of columns in B and C
 * @param k       Number of columns in A and rows in B
 * @return        0 on success, non-zero on error
 */
int tiled_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k);

/**
 * Compute C = A * B using register-blocked GEMM kernel
 * 
 * @param h_a     Input matrix A on host (row-major, m x k)
 * @param h_b     Input matrix B on host (row-major, k x n)
 * @param h_c     Output matrix C on host (row-major, m x n)
 * @param m       Number of rows in A and C
 * @param n       Number of columns in B and C
 * @param k       Number of columns in A and rows in B
 * @return        0 on success, non-zero on error
 */
int register_blocked_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k);

#ifdef __cplusplus
}
#endif
