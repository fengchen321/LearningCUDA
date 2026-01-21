/**
 * @file nvtx_demo.cu
 * @brief NVTX 标记测试 - 测试 NVTX 范围
 *
 * nsys 测试命令:
 *   nsys profile --stats=true --nvtx -o nvtx_report ./nvtx_demo
 *   nsys profile --nvtx --nvtx-include="Region A" -o nvtx_include ./nvtx_demo
 *   nsys profile --nvtx --nvtx-exclude="Region B" -o nvtx_exclude ./nvtx_demo
 *
 * NVTX v3 API:
 *   nvtx3::mark() - 标记点
 *   nvtx3::scoped_range() - 自动管理的范围
 */
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024)
#define BLOCK_SIZE 256

__global__ void kernel_a(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 50; i++) {
            data[idx] += data[idx] * 0.01f;
        }
    }
}

__global__ void kernel_b(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 100; i++) {
            data[idx] = sqrtf(data[idx] + 1.0f);
        }
    }
}

__global__ void kernel_c(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 30; i++) {
            data[idx] = data[idx] * 2.0f;
        }
    }
}

int main(int argc, char** argv) {
    float *d_data;
    size_t size = N * sizeof(float);

    // 分配设备内存
    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 1, size);

    int iterations = 5;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    // 标记整个外层循环 (使用 NVTX3_FUNC_RANGE 宏)
    NVTX3_FUNC_RANGE();

    for (int iter = 0; iter < iterations; iter++) {
        // Region A: 主要计算 (使用 scoped_range)
        {
            nvtx3::scoped_range range("Region A");
            int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_a<<<blocks, BLOCK_SIZE>>>(d_data, N);
            cudaDeviceSynchronize();
        }

        // Region B: 第二计算
        {
            nvtx3::scoped_range range("Region B");
            int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_b<<<blocks, BLOCK_SIZE>>>(d_data, N);
            cudaDeviceSynchronize();
        }

        // Region C: 额外计算
        {
            nvtx3::scoped_range range("Region C");
            int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_c<<<blocks, BLOCK_SIZE>>>(d_data, N);
            cudaDeviceSynchronize();
        }
    }

    // 清理
    cudaFree(d_data);

    printf("NVTX demo completed.\n");
    return 0;
}
