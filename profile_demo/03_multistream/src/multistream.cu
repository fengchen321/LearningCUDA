/**
 * @file multistream.cu
 * @brief 多流测试 - 测试异步执行和流间同步
 *
 * nsys 测试命令:
 *   nsys profile --stats=true -o multistream_report ./multistream
 *   nsys profile -t cuda -o multistream_cuda ./multistream
 *
 * 测试场景:
 *   - 多个 CUDA 流并行执行
 *   - 流间同步 (cudaStreamSynchronize, cudaEvent)
 *   - 默认流行为
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (8 * 1024 * 1024)
#define NUM_STREAMS 4
#define ITERATIONS 10

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 100; i++) {  // 增加计算量
            c[idx] = a[idx] + b[idx] + c[idx] * 0.01f;
        }
    }
}

int main(int argc, char** argv) {
    float *h_a, *h_b, *h_c;
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t start, stop;

    size_t size = N * sizeof(float);

    // 分配主机内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 创建流和事件
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_a[i], size);
        cudaMalloc(&d_b[i], size);
        cudaMalloc(&d_c[i], size);
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 测试多流执行
    printf("Testing %d streams...\n", NUM_STREAMS);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        // 并行提交到所有流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaMemcpyAsync(d_a[i], h_a, size, cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(d_b[i], h_b, size, cudaMemcpyHostToDevice, streams[i]);

            int blocks = (N + 255) / 256;
            vector_add<<<blocks, 256, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], N);

            cudaMemcpyAsync(h_c, d_c[i], size, cudaMemcpyDeviceToHost, streams[i]);
        }

        // 等待所有流完成
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
        }
    }

    // 清理
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_a);
    free(h_b);
    free(h_c);

    printf("Multi-stream test completed.\n");
    return 0;
}
