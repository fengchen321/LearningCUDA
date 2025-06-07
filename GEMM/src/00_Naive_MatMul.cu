#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v00.  Naive Matrix multiplication kernel called by gemm_v00

template <typename T>
__global__ void gemm_v00(const Matrix<T> A, const Matrix<T> B, Matrix<T> C)
{
    // Each thread computr one element of C
    // by accmulating results into Cvalue
    T Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t e = 0; e < A.width; ++e) {
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    }
    C.elements[row * C.width + col] = Cvalue;
}

template <typename T>
void launch_gemm_kernel_v00(const Matrix<T> A, const Matrix<T> B, Matrix<T> C, cudaStream_t stream)
{
    dim3 const block_dim{16U, 16U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(B.width) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(A.height) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(A, B, C);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v00<float>(const Matrix<float>, const Matrix<float>, Matrix<float>, cudaStream_t);

template void launch_gemm_kernel_v00<double>(const Matrix<double>, const Matrix<double>, Matrix<double>, cudaStream_t);