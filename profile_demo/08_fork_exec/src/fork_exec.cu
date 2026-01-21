/**
 * @file fork_exec.cu
 * @brief fork-exec 测试 - 测试子进程跟踪
 *
 * nsys 测试命令:
 *   nsys profile --stats=true --trace-fork-before-exec=true -o fork_report ./fork_exec
 *
 * 测试场景:
 *   - fork() 创建子进程
 *   - exec() 加载新程序
 *   - 父子进程 CUDA 上下文继承
 *
 * 注意: 这个 demo 需要在支持 fork 的系统上运行 (Linux)
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

#define N (256 * 1024)

__global__ void simple_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 100; i++) {
            data[idx] += data[idx] * 0.01f;
        }
    }
}

void child_process() {
    printf("[Child] PID: %d, PPID: %d\n", getpid(), getppid());

    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 1, N);

    int blocks = (N + 255) / 256;
    simple_kernel<<<blocks, 256>>>(d_data, N);
    cudaDeviceSynchronize();

    printf("[Child] CUDA computation done\n");

    cudaFree(d_data);
}

int main(int argc, char** argv) {
    printf("[Parent] PID: %d\n", getpid());

    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 1, N);

    // 父进程先执行一些 CUDA 操作
    printf("[Parent] Initial CUDA work\n");
    int blocks = (N + 255) / 256;
    simple_kernel<<<blocks, 256>>>(d_data, N);
    cudaDeviceSynchronize();

    // fork 测试
    printf("[Parent] Forking child process...\n");
    pid_t pid = fork();

    if (pid == 0) {
        // 子进程
        // 选项 1: 不执行 exec，直接用 CUDA
        // child_process();

        // 选项 2: exec 加载新程序 (这里用简单的 sleep 演示)
        printf("[Child] Executing new program via exec...\n");
        execl("/bin/sleep", "sleep", "1", NULL);
        // 如果 exec 失败
        perror("exec failed");
        exit(1);
    } else if (pid > 0) {
        // 父进程等待
        int status;
        waitpid(pid, &status, 0);
        printf("[Parent] Child exited with status %d\n", WEXITSTATUS(status));

        // 继续父进程的 CUDA 工作
        printf("[Parent] Continuing CUDA work\n");
        simple_kernel<<<blocks, 256>>>(d_data, N);
        cudaDeviceSynchronize();
    } else {
        perror("fork failed");
    }

    cudaFree(d_data);

    printf("[Parent] Fork-exec demo completed.\n");
    return 0;
}
