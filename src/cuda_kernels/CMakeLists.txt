cmake_minimum_required(VERSION 3.18)
project(gemm_cuda_kernels CUDA CXX)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# Create the kernel library
add_library(gemm_kernels SHARED
    kernels.cu
    gemm_c_interface.cu
)

# Set include directories
target_include_directories(gemm_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set CUDA architecture (adjust based on your GPU)
# Common options: 60 (P100), 70 (V100), 75 (RTX 2080), 80 (A100), 86 (RTX 3080)
set_target_properties(gemm_kernels PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

# Optional: Create a static library variant
add_library(gemm_kernels_static STATIC
    kernels.cu
    gemm_c_interface.cu
)

target_include_directories(gemm_kernels_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(gemm_kernels_static PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

# Print configuration info
message(STATUS "CUDA Toolkit Version: ${CMAKE_CUDA_TOOLKIT_VERSION}")
message(STATUS "CUDA Architectures: 60;70;75;80;86")
message(STATUS "Built shared library: gemm_kernels")
message(STATUS "Built static library: gemm_kernels_static")
