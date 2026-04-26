#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <string>

// ============================================================================
// CUDA ERROR CHECKING UTILITIES
// ============================================================================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// ============================================================================
// KERNEL LAUNCH WRAPPER FUNCTIONS
// ============================================================================

#include "kernels.h"

/**
 * Launch naive GEMM kernel
 * Computes C = A * B for matrices of size (m x k) * (k x n) = (m x n)
 */
inline void launch_naive_gemm(const float *d_a, const float *d_b, float *d_c,
                              int m, int n, int k) {
    dim3 block(NAIVE_BLOCK_X, NAIVE_BLOCK_Y);
    dim3 grid((n + NAIVE_BLOCK_X - 1) / NAIVE_BLOCK_X,
              (m + NAIVE_BLOCK_Y - 1) / NAIVE_BLOCK_Y);
    
    naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CHECK_CUDA(cudaGetLastError());
}

/**
 * Launch tiled GEMM kernel
 */
inline void launch_tiled_gemm(const float *d_a, const float *d_b, float *d_c,
                              int m, int n, int k) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (m + TILE_SIZE - 1) / TILE_SIZE);
    
    tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CHECK_CUDA(cudaGetLastError());
}

/**
 * Launch register-blocked GEMM kernel
 */
inline void launch_register_blocked_gemm(const float *d_a, const float *d_b, float *d_c,
                                         int m, int n, int k) {
    dim3 block(RB_BLOCK_THREADS_X, RB_BLOCK_THREADS_Y);
    dim3 grid((n + RB_BLOCK_TILE_N - 1) / RB_BLOCK_TILE_N,
              (m + RB_BLOCK_TILE_M - 1) / RB_BLOCK_TILE_M);
    
    register_blocked_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CHECK_CUDA(cudaGetLastError());
}

/**
 * Launch high-performance GEMM kernel (128x128 block tile, 8x8 thread tile)
 */
inline void launch_hpc_gemm(const float *d_a, const float *d_b, float *d_c,
                             int m, int n, int k) {
    dim3 block(HPC_THREADS_X, HPC_THREADS_Y);
    dim3 grid((n + HPC_BLOCK_N - 1) / HPC_BLOCK_N,
              (m + HPC_BLOCK_M - 1) / HPC_BLOCK_M);
    hpc_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CHECK_CUDA(cudaGetLastError());
}

/**
 * Launch ultra high-performance GEMM kernel (128x128 block tile, 8x8 thread tile, K-tile=32)
 */
inline void launch_ultra_gemm(const float *d_a, const float *d_b, float *d_c,
                               int m, int n, int k) {
    dim3 block(ULTRA_THREADS_X, ULTRA_THREADS_Y);
    dim3 grid((n + ULTRA_BLOCK_N - 1) / ULTRA_BLOCK_N,
              (m + ULTRA_BLOCK_M - 1) / ULTRA_BLOCK_M);
    ultra_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CHECK_CUDA(cudaGetLastError());
}

/**
 * Synchronize device and check for errors
 */
inline void sync_device() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

/**
 * Allocate device memory
 */
inline float* malloc_device(size_t num_elements) {
    float *ptr;
    CHECK_CUDA(cudaMalloc(&ptr, num_elements * sizeof(float)));
    return ptr;
}

/**
 * Free device memory
 */
inline void free_device(float *ptr) {
    CHECK_CUDA(cudaFree(ptr));
}

/**
 * Copy data from host to device
 */
inline void copy_to_device(float *d_ptr, const float *h_ptr, size_t num_elements) {
    CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, num_elements * sizeof(float), cudaMemcpyHostToDevice));
}

/**
 * Copy data from device to host
 */
inline void copy_to_host(float *h_ptr, const float *d_ptr, size_t num_elements) {
    CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
}

#endif // GEMM_UTILS_H
