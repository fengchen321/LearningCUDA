/**
 * @file memory_test.cu
 * @brief 内存传输测试 - 测试 H2D/D2H/P2P 带宽
 *
 * nsys 测试命令:
 *   nsys profile --stats=true -o memory_report ./memory_test
 *   nsys profile -t cuda,osrt -o memory_trace ./memory_test
 *
 * 测试场景:
 *   - 多次 H2D/D2H 传输
 *   - 页面锁定内存 (pinned memory)
 *   - 设备间 P2P 传输 (多GPU)
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE_1MB (1024 * 1024)
#define NUM_ITERATIONS 100

// 使用 pinned memory 的传输
__global__ void memory_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size / sizeof(float); i += stride) {
        data[i] *= 2.0f;
    }
}

int main(int argc, char** argv) {
    float *h_pageable, *h_pinned;
    float *d_data;
    size_t size = 64 * SIZE_1MB;  // 64MB

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    // 分配 pageable 内存
    h_pageable = (float*)malloc(size);

    // 分配 pinned 内存
    cudaMallocHost(&h_pinned, size);

    // 初始化
    for (size_t i = 0; i < size / sizeof(float); i++) {
        h_pinned[i] = (float)i / (size / sizeof(float));
    }

    // 设备内存分配
    cudaMalloc(&d_data, size);

    // 测试 1: Pageable 内存传输
    printf("Testing pageable memory...\n");
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_pageable, d_data, size, cudaMemcpyDeviceToHost);
    }

    // 测试 2: Pinned 内存传输
    printf("Testing pinned memory...\n");
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost);
    }

    // 测试 3: 设备上多次访问
    printf("Testing device memory access...\n");
    int blocks = 64;
    int threads = 256;
    for (int i = 0; i < 10; i++) {
        memory_kernel<<<blocks, threads>>>(d_data, size);
        cudaDeviceSynchronize();
    }

    // 清理
    cudaFree(d_data);
    cudaFreeHost(h_pinned);
    free(h_pageable);

    printf("Memory test completed.\n");
    return 0;
}
