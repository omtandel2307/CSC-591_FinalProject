#include "gemm_c_interface.h"
#include "kernels_utils.h"
#include <string.h>

/**
 * Internal helper: allocate GPU memory, copy host to device, launch kernel, copy back
 */
static int gemm_helper(const float *h_a, const float *h_b, float *h_c, 
                       int m, int n, int k,
                       void (*kernel_launcher)(const float*, const float*, float*, int, int, int)) {
    try {
        // Allocate device memory
        float *d_a = malloc_device(m * k);
        float *d_b = malloc_device(k * n);
        float *d_c = malloc_device(m * n);

        // Copy input matrices to device
        copy_to_device(d_a, h_a, m * k);
        copy_to_device(d_b, h_b, k * n);

        // Launch kernel
        kernel_launcher(d_a, d_b, d_c, m, n, k);

        // Synchronize and check for errors
        sync_device();

        // Copy result back to host
        copy_to_host(h_c, d_c, m * n);

        // Free device memory
        free_device(d_a);
        free_device(d_b);
        free_device(d_c);

        return 0;  // Success
    } catch (...) {
        return -1;  // Error
    }
}

/**
 * Wrapper: Naive GEMM
 */
int naive_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k) {
    return gemm_helper(h_a, h_b, h_c, m, n, k, launch_naive_gemm);
}

/**
 * Wrapper: Tiled GEMM
 */
int tiled_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k) {
    return gemm_helper(h_a, h_b, h_c, m, n, k, launch_tiled_gemm);
}

/**
 * Wrapper: Register-blocked GEMM
 */
int register_blocked_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k) {
    return gemm_helper(h_a, h_b, h_c, m, n, k, launch_register_blocked_gemm);
}
