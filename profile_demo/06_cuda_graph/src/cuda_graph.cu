/**
 * @file cuda_graph.cu
 * @brief CUDA Graphs 测试 - 测试图捕获和执行
 *
 * nsys 测试命令:
 *   nsys profile --stats=true --cuda-graph-trace=graph -o graph_report ./cuda_graph
 *   nsys profile --cuda-graph-trace=node -o graph_node ./cuda_graph
 *
 * CUDA Graphs 功能:
 *   - cudaStreamBeginCapture() - 开始捕获流
 *   - cudaStreamEndCapture() - 结束捕获并获取图
 *   - cudaGraphInstantiate() - 实例化图
 *   - cudaGraphLaunch() - 执行图
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024)
#define ITERATIONS 10

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 50; i++) {
            c[idx] = a[idx] + b[idx] + c[idx] * 0.01f;
        }
    }
}

int main(int argc, char** argv) {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // 分配主机内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaStreamCreate(&stream);

    int blocks = (N + 255) / 256;

    // 方法: 使用 Stream Capture API (更简单且兼容)
    printf("Testing CUDA Graph with stream capture...\n");

    // 开始捕获流到图
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // 这些操作会被捕获到图中
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);
    vector_add<<<blocks, 256, 0, stream>>>(d_a, d_b, d_c, N);
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);

    // 结束捕获，获取图
    cudaStreamEndCapture(stream, &graph);

    // 实例化图
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // 多次执行图
    for (int i = 0; i < ITERATIONS; i++) {
        cudaGraphLaunch(graphExec, stream);
    }

    // 等待完成
    cudaStreamSynchronize(stream);

    // 清理图
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    // 传统方法对比
    printf("Testing traditional execution...\n");
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    for (int i = 0; i < ITERATIONS; i++) {
        cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);
        vector_add<<<blocks, 256, 0, stream2>>>(d_a, d_b, d_c, N);
        cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream2);
    }
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream2);

    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    printf("CUDA Graph demo completed.\n");
    return 0;
}
