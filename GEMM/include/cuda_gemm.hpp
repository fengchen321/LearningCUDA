#ifndef CUDA_GEMM_HPP
#define CUDA_GEMM_HPP

#include <cuda_runtime.h>
template <typename T>
struct Matrix{
    int width;
    int height;
    int stride;
    float* elements;
};

template <typename T>
void launch_gemm_kernel_v00(const Matrix<T> A, const Matrix<T> B, Matrix<T> C, cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v01(const Matrix<T> A, const Matrix<T> B, Matrix<T> C, cudaStream_t stream);

#endif