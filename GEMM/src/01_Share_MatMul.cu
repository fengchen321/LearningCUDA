#include <cuda_fp16.h>
#include <iostream>
#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v01. Shared Memory
// Get a matrix element
#define BLOCK_SIZE 16
template <typename T>
__device__ T GetElement(const Matrix<T> A, int row, int col) {
    if (row >= A.height || col >= A.width) return 0;
    return A.elements[row * A.stride + col];
}

// Set a matrix element
template <typename T>
__device__ void SetElement(Matrix<T>& A, int row, int col, T value) {
    if (row < A.height && col < A.width) {
        A.elements[row * A.stride + col] = value;
    }
}

// Get the BLOCK_SIZE * BLOCK_SIZE sub-matrix Asub of A that is 
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
template <typename T>
__device__ Matrix<T> GetSubMatrix(const Matrix<T> A, int row, int col) {
    Matrix<T> Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

template <typename T>
__global__ void gemm_v01(const Matrix<T> A, const Matrix<T> B, Matrix<T> C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix<T> Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub by accumulating results into Cvalue
    T Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are required to compute Csub
    // Multiply each pair of sub-matrices together and accumulate the results
    for (int m = 0; m < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        // Get sub-matrix Asub of A
        Matrix<T> Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix<T> Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        if (row < Asub.height && col < Asub.width) {
            As[row][col] = GetElement(Asub, row, col);
        } else {
            As[row][col] = 0;
        }

        if (row < Bsub.height && col < Bsub.width) {
            Bs[row][col] = GetElement(Bsub, row, col);
        } else {
            Bs[row][col] = 0;
        }

        // Synchronize to make sure the sub-matrices are loaded before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory, each thread writes one element
     if (row < Csub.height && col < Csub.width) {
        SetElement(Csub, row, col, Cvalue);
    }
}

template <typename T>
void launch_gemm_kernel_v01(const Matrix<T> A, const Matrix<T> B, Matrix<T> C, cudaStream_t stream)
{
    if (A.width != B.height || C.height != A.height || C.width != B.width) {
        std::cerr << "Error: Matrix dimensions mismatch!" << std::endl;
        std::cerr << "A: " << A.height << "x" << A.width << std::endl;
        std::cerr << "B: " << B.height << "x" << B.width << std::endl;
        std::cerr << "C: " << C.height << "x" << C.width << std::endl;
        return;
    }
    dim3 const block_dim{BLOCK_SIZE, BLOCK_SIZE, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(B.width) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(A.height) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v01<T><<<grid_dim, block_dim, 0U, stream>>>(A, B, C);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v01<float>(const Matrix<float>, const Matrix<float>, Matrix<float>, cudaStream_t);
template void launch_gemm_kernel_v01<double>(const Matrix<double>, const Matrix<double>, Matrix<double>, cudaStream_t);