/**
 * @file basic.cu
 * @brief 基础向量加法 - 测试 nsys 基础采集
 *
 * nsys 测试命令:
 *   nsys profile --stats=true -o basic_report ./basic
 *   nsys profile -t cuda,nvtx,osrt -o basic_trace ./basic
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024 * 1024
#define BLOCK_SIZE 256

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void init_data(float* h_a, float* h_b, int n) {
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char** argv) {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // 分配主机内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 初始化数据
    init_data(h_a, h_b, N);

    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 多次迭代测试
    int iterations = 10;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    for (int iter = 0; iter < iterations; iter++) {
        // H2D 传输
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

        // 内核执行
        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        // D2H 传输
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    }

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    printf("Basic vector add completed. %d iterations.\n", iterations);
    return 0;
}
