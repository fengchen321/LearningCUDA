cmake_minimum_required(VERSION 3.23)

project(GEMM
    VERSION 0.0.1
    LANGUAGES CXX CUDA
    DESCRIPTION "CUDA GEMM Library"
)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Dependencies
# =============================================================================
find_package(CUDAToolkit REQUIRED)

# =============================================================================
# Include Directories
# =============================================================================
set(GEMM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# =============================================================================
# Compiler Options
# =============================================================================
set(GEMM_COMMON_WARNINGS
    -Wall
    -Wextra
    # -Wpedantic
)

set(GEMM_CUDA_WARNINGS
    -Xcompiler=${GEMM_COMMON_WARNINGS}
)

# Debug specific options - only apply in Debug build
set(GEMM_DEBUG_OPTIONS
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(GEMM_CUDA_ARCHITECTURES all-major)

# =============================================================================
# Source Files
# =============================================================================
set(GEMM_SOURCES
    src/cuda_gemm_utils.cu
    src/00_Naive_MatMul.cu
    src/01_Share_MatMul.cu
)

# =============================================================================
# Library Target: cuda_gemm
# =============================================================================
add_library(cuda_gemm SHARED ${GEMM_SOURCES})

target_include_directories(cuda_gemm PUBLIC ${GEMM_INCLUDE_DIR})
target_link_libraries(cuda_gemm PRIVATE CUDA::cublas)

set_target_properties(cuda_gemm PROPERTIES
    CUDA_ARCHITECTURES ${GEMM_CUDA_ARCHITECTURES}
    OUTPUT_NAME cuda_gemm
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

target_compile_options(cuda_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${GEMM_COMMON_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${GEMM_CUDA_WARNINGS}>
    ${GEMM_DEBUG_OPTIONS}
)

# =============================================================================
# Executable Target: profile_cuda_gemm_fp32
# =============================================================================
add_executable(profile_cuda_gemm_fp32 src/profile_cuda_gemm_fp32.cu)

target_include_directories(profile_cuda_gemm_fp32 PUBLIC ${GEMM_INCLUDE_DIR})
target_link_libraries(profile_cuda_gemm_fp32 PRIVATE
    cuda_gemm
    CUDA::cublas
)

set_target_properties(profile_cuda_gemm_fp32 PROPERTIES
    CUDA_ARCHITECTURES ${GEMM_CUDA_ARCHITECTURES}
    OUTPUT_NAME profile_cuda_gemm_fp32
)

target_compile_options(profile_cuda_gemm_fp32 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${GEMM_COMMON_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${GEMM_CUDA_WARNINGS}>
    ${GEMM_DEBUG_OPTIONS}
)