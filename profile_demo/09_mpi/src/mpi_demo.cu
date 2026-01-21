/**
 * @file mpi_demo.cu
 * @brief MPI + CUDA 测试 - 测试多进程 GPU 计算
 *
 * nsys 测试命令:
 *   nsys profile --stats=true --mpi=openmpi -o mpi_report ./mpi_demo
 *   nsys profile --mpi=mpich -o mpi_mpich ./mpi_demo
 *
 * 编译:
 *   CUDAARCHS=$(cuda_compute_cap) mpicc -o mpi_demo mpi_demo.cu -lcudart
 *   或使用 cmake
 *
 * 运行:
 *   mpirun -np 4 ./mpi_demo
 */
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024)
#define TAG_DATA 1

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 50; i++) {
            c[idx] = a[idx] + b[idx] + c[idx] * 0.01f;
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size_per_proc = N * sizeof(float);

    // 初始化 MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("MPI+CUDA Demo with %d processes\n", size);
    }

    // 每个进程分配数据
    h_a = (float*)malloc(size_per_proc);
    h_b = (float*)malloc(size_per_proc);
    h_c = (float*)malloc(size_per_proc);

    // 初始化本地数据
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rank;
        h_b[i] = (float)(i % 256);
    }

    // 分配设备内存
    cudaSetDevice(rank % 2);  // 假设每个节点有 2 个 GPU
    cudaMalloc(&d_a, size_per_proc);
    cudaMalloc(&d_b, size_per_proc);
    cudaMalloc(&d_c, size_per_proc);

    // H2D 传输
    cudaMemcpy(d_a, h_a, size_per_proc, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_per_proc, cudaMemcpyHostToDevice);

    // GPU 计算
    int blocks = (N + 255) / 256;
    vector_add<<<blocks, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // D2H 传输
    cudaMemcpy(h_c, d_c, size_per_proc, cudaMemcpyDeviceToHost);

    // MPI 收集结果 (rank 0 接收)
    if (rank == 0) {
        float* h_all = (float*)malloc(size_per_proc * size);

        MPI_Gather(h_c, N, MPI_FLOAT, h_all, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // 验证结果
        float sum = 0;
        for (int p = 0; p < size; p++) {
            for (int i = 0; i < 10; i++) {
                sum += h_all[p * N + i];
            }
        }
        printf("Rank 0: Sample sum = %f\n", sum);

        free(h_all);
    } else {
        MPI_Gather(h_c, N, MPI_FLOAT, NULL, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    cudaDeviceSynchronize();

    MPI_Finalize();

    if (rank == 0) {
        printf("MPI+CUDA demo completed.\n");
    }

    return 0;
}
