# CUDA GEMM Kernels

This directory contains optimized CUDA implementations of three GEMM (General Matrix Multiply) kernels:

1. **Naive GEMM** - Basic implementation with no optimization
2. **Tiled GEMM** - Uses shared memory for improved data reuse  
3. **Register-blocked GEMM** - Each thread computes multiple output elements in registers

## Files

- `kernels.h` - Kernel declarations and configuration constants
- `kernels.cu` - CUDA kernel implementations
- `kernels_utils.h` - Helper functions for memory management and kernel launching
- `gemm_c_interface.h` - C-style interface for Python integration
- `gemm_c_interface.cu` - C interface implementation
- `CMakeLists.txt` - Build configuration

## Building

### Option 1: Using CMake (Recommended)

```bash
cd src/cuda_kernels
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

This creates:
- `gemm_kernels.dll` (shared library on Windows) / `libgemm_kernels.so` (on Linux)
- `gemm_kernels_static.lib` / `libgemm_kernels_static.a` (static library)

### Option 2: Direct nvcc compilation (Advanced)

```bash
# Compile to object files
nvcc -c -O3 -arch=sm_75 kernels.cu -o kernels.o
nvcc -c -O3 -arch=sm_75 gemm_c_interface.cu -o gemm_c_interface.o

# Link into shared library
nvcc -shared -O3 -arch=sm_75 kernels.o gemm_c_interface.o -o gemm_kernels.dll
```

Replace `sm_75` with your GPU's compute capability:
- `sm_60` - Pascal (P100, P40)
- `sm_70` - Volta (V100)
- `sm_75` - Turing (RTX 2080, RTX 2070)
- `sm_80` - Ampere (A100, RTX 3090)
- `sm_86` - Ampere (RTX 3080, RTX 3060)

## API

### C Functions

All functions take host-memory pointers and perform device allocation/deallocation internally:

```c
// Naive GEMM: global memory only
int naive_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k);

// Tiled GEMM: shared memory optimization
int tiled_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k);

// Register-blocked GEMM: register + shared memory optimization  
int register_blocked_gemm(const float *h_a, const float *h_b, float *h_c, int m, int n, int k);
```

**Parameters:**
- `h_a` - Input matrix A (m × k), row-major in host memory
- `h_b` - Input matrix B (k × n), row-major in host memory
- `h_c` - Output matrix C (m × n), row-major in host memory
- `m` - Number of rows in A and C
- `n` - Number of columns in B and C
- `k` - Inner dimension (columns of A, rows of B)

**Returns:** 0 on success, -1 on error

## Python Usage

Using ctypes to call the compiled library:

```python
import ctypes
import numpy as np

# Load the compiled library
lib = ctypes.CDLL('./src/cuda_kernels/build/gemm_kernels.dll')  # Windows
# lib = ctypes.CDLL('./src/cuda_kernels/build/libgemm_kernels.so')  # Linux

# Define function signatures
lib.naive_gemm.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.naive_gemm.restype = ctypes.c_int

# Prepare data (row-major, C-contiguous)
m, n, k = 1024, 1024, 1024
A = np.random.randn(m, k).astype(np.float32)
B = np.random.randn(k, n).astype(np.float32)
C = np.zeros((m, n), dtype=np.float32)

# Call kernel
status = lib.naive_gemm(
    A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k)
)

if status != 0:
    print("Error running kernel")
else:
    # C now contains A @ B
    print(f"Result shape: {C.shape}")
```

## Kernel Architecture

### Naive Kernel (naive_gemm_kernel)
- Block size: 16×16 threads
- Each thread computes one output element C[row, col]
- All accesses to global memory
- Low arithmetic intensity, high memory bandwidth requirement

### Tiled Kernel (tiled_gemm_kernel)  
- Block size: 16×16 threads
- Divides matrices into 16×16 tiles
- Loads tiles into shared memory
- Computes partial products for each tile
- Better data reuse than naive

### Register-blocked Kernel (register_blocked_gemm_kernel)
- Block size: 16×16 threads
- Each thread computes 2×2 output tile (4 elements)
- Tiles in K dimension using shared memory (16-wide)
- Accumulates in registers (high arithmetic intensity)
- Best performance with minimal shared memory usage

## Performance Considerations

1. **Memory Coalescing**: All kernels use row-major layout for coalesced memory access
2. **Shared Memory**: Tiled and register-blocked kernels reduce global memory bandwidth
3. **Occupancy**: Register-blocked may have lower occupancy due to register usage
4. **Arithmetic Intensity**: Increases from naive → tiled → register-blocked

## Compilation Notes

- Requires CUDA Toolkit (version 11.0+)
- C++17 standard required
- Make sure `nvcc` is in your PATH
- CMake 3.18+ for CUDA support
