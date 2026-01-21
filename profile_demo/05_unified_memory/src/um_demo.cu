/**
 * @file um_demo.cu
 * @brief 统一内存测试 - 测试 UM 页错误和迁移
 *
 * nsys 测试命令:
 *   nsys profile --stats=true --cuda-um-cpu-page-faults=true \
 *                --cuda-um-gpu-page-faults=true -o um_report ./um_demo
 *
 * 测试场景:
 *   - 频繁的 CPU/GPU 访问切换
 *   - 页面错误计数
 *   - 首次访问延迟
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (16 * 1024 * 1024)  // 64MB

// 统一内存版本
__global__ void compute_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += data[idx] * 0.001f;
        }
        data[idx] = sum;
    }
}

int main(int argc, char** argv) {
    float *data_managed;

    // 使用统一内存分配
    cudaMallocManaged(&data_managed, N * sizeof(float));

    // 预热 GPU
    int blocks = (N + 255) / 256;
    compute_kernel<<<blocks, 256>>>(data_managed, N);
    cudaDeviceSynchronize();

    // 测试 1: CPU 写入，然后 GPU 读取
    printf("Test 1: CPU write -> GPU read\n");
    for (int i = 0; i < N; i += 1024) {
        data_managed[i] = (float)i;
    }
    compute_kernel<<<blocks, 256>>>(data_managed, N);
    cudaDeviceSynchronize();

    // 测试 2: GPU 写入，然后 CPU 读取
    printf("Test 2: GPU write -> CPU read\n");
    compute_kernel<<<blocks, 256>>>(data_managed, N);
    cudaDeviceSynchronize();

    float sum = 0.0f;
    for (int i = 0; i < N; i += 1024) {
        sum += data_managed[i];
    }
    printf("CPU read sum: %f\n", sum);

    // 测试 3: 频繁的 CPU/GPU 切换访问
    printf("Test 3: Frequent CPU/GPU switching\n");
    for (int iter = 0; iter < 10; iter++) {
        // CPU 修改一小部分
        for (int i = 0; i < 4096; i++) {
            data_managed[i] += 1.0f;
        }

        // GPU 读取并计算
        compute_kernel<<<blocks, 256>>>(data_managed, N);
        cudaDeviceSynchronize();
    }

    // 测试 4: 随机访问模式
    printf("Test 4: Random access pattern\n");
    for (int iter = 0; iter < 5; iter++) {
        for (int i = 0; i < N; i += 4096) {
            data_managed[i] *= 2.0f;
        }
        compute_kernel<<<blocks, 256>>>(data_managed, N);
        cudaDeviceSynchronize();
    }

    // 清理
    cudaFree(data_managed);

    printf("Unified memory demo completed.\n");
    return 0;
}
