cmake_minimum_required(VERSION 3.24)

project(GEMM VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 直接设置正确的头文件路径（推荐）
set(CUDA_GEMM_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/GEMM/include)

# 添加所有头文件（可选）
file(GLOB CUDA_GEMM_HEADERS ${CUDA_GEMM_INCLUDE_DIRS}/*.hpp ${CUDA_GEMM_INCLUDE_DIRS}/*.cuh)

# 编译 CUDA 库
add_library(
    cuda_gemm 
    SHARED 
    src/cuda_gemm_utils.cu 
    src/00_Naive_MatMul.cu
    src/01_Share_MatMul.cu
)

# 添加头文件目录
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})

# 设置 CUDA 架构
set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES native)

# 编译可执行文件
add_executable(profile_cuda_gemm_fp32 src/profile_cuda_gemm_fp32.cu)
target_include_directories(profile_cuda_gemm_fp32 PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})
target_link_libraries(profile_cuda_gemm_fp32 PRIVATE cuda_gemm CUDA::cublas) 
set_target_properties(profile_cuda_gemm_fp32 PROPERTIES CUDA_ARCHITECTURES native)
