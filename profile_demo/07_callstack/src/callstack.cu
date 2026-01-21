/**
 * @file callstack.cu
 * @brief 调用链测试 - 测试 CPU 回溯采样
 *
 * nsys 测试命令:
 *   nsys profile --stats=true --sample=process-tree --backtrace=lbr -o lbr_report ./callstack
 *   nsys profile --sample=process-tree --backtrace=dwarf -o dwarf_report ./callstack
 *
 * 回溯方法:
 *   - lbr (Last Branch Record): 快速，但信息有限
 *   - fp (Frame Pointer): 需要 -O0 编译
 *   - dwarf: 最完整，需要 debug info
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1024 * 1024)
#define BLOCK_SIZE 256

// 基础计算核函数
__global__ void compute_kernel(float* data, int n, int work_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < work_factor; i++) {
            sum += sinf(data[idx] * 0.01f + i) * cosf(data[idx] * 0.02f + i);
        }
        data[idx] = sum;
    }
}

// 包装函数调用链
void wrapper_level3(float* data, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_kernel<<<blocks, BLOCK_SIZE>>>(data, n, 200);
    cudaDeviceSynchronize();
}

void wrapper_level2(float* data, int n) {
    wrapper_level3(data, n);
}

void wrapper_level1(float* data, int n) {
    wrapper_level2(data, n);
}

void top_wrapper(float* data, int n) {
    wrapper_level1(data, n);
}

int main(int argc, char** argv) {
    float *d_data;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 1, size);

    int iterations = 5;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    // 使用多层 C++ 包装函数调用 CUDA
    for (int iter = 0; iter < iterations; iter++) {
        printf("Iteration %d/%d\n", iter + 1, iterations);

        // 通过 C++ 包装函数调用，增加 CPU 调用链深度
        top_wrapper(d_data, N);

        // 直接启动
        int blocks = (N + 255) / 256;
        compute_kernel<<<blocks, BLOCK_SIZE>>>(d_data, N, 100);
        cudaDeviceSynchronize();
    }

    cudaFree(d_data);

    printf("Callstack demo completed.\n");
    return 0;
}
