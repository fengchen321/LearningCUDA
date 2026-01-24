/**
 * @file nvtx_demo.cu
 * @brief NVTX v1 API 全面测试demo (兼容旧版本)
 *
 * 编译: nvcc -o nvtx_demo nvtx_demo.cu -lnvToolsExt
 *
 * nsys 测试命令:
 *   nsys profile --stats=true --nvtx -o nvtx_report ./nvtx_demo --mode=all
 *   nsys profile --nvtx --nvtx-include="Domain A@*" -o nvtx_include ./nvtx_demo --mode=domain
 *   nsys profile --nvtx --nvtx-exclude="Region B" -o nvtx_exclude ./nvtx_demo --mode=range
 *
 * 支持的测试模式 (--mode):
 *   basic      - 基础标记和范围
 *   domain     - 自定义域测试
 *   nested     - 嵌套范围测试
 *   mark       - 标记点测试
 *   func-auto  - 函数级自动范围(NVTX3_FUNC_RANGE)
 *   all        - 所有功能
 *   help       - 显示帮助信息
 */
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(error)                                   \
{                                                           \
  cudaError_t localError = error;                           \
  if ((localError != cudaSuccess) &&                        \
      (localError != cudaErrorPeerAccessAlreadyEnabled)) {  \
    printf("error: '%s'(%d) from %s at %s:%d\n",            \
           cudaGetErrorString(localError),                  \
           localError, #error, __FUNCTION__, __LINE__);     \
    exit(1);                                                \
  }                                                         \
}

#define N (1024 * 1024)
#define BLOCK_SIZE 256

static float *g_d_b = nullptr;

/* ---------------- help ---------------- */
void print_help(const char* program_name) {
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  --mode <MODE>     测试模式 (默认: basic)\n");
    printf("  --iter <N>        迭代次数 (默认: 5)\n");
    printf("  --help, -h        显示帮助信息\n\n");
    printf("Available modes:\n");
    printf("  basic     - 基础标记(nvtxMark / nvtxRange)\n");
    printf("  domain    - 自定义域测试(nvtxDomain*)\n");
    printf("  nested    - 嵌套范围测试\n");
    printf("  mark      - 标记点测试\n");
    printf("  func-auto - 函数级自动范围(NVTX3_FUNC_RANGE)\n");
    printf("  all       - 所有功能测试\n\n");
}

/* ---------------- kernels ---------------- */
__global__ void vector_add(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 10; i++)
            out[idx] = a[idx] + b[idx] * 0.01f;
    }
}

__global__ void vector_mul(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 20; i++) {
            out[idx] = a[idx] * b[idx] * 0.5f;
        }
    }
}

__global__ void vector_scale(float* out, const float* a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 15; i++) {
            out[idx] = a[idx] * scalar;
        }
    }
}

// ============================================================================
// 基础标记和范围测试
// ============================================================================
void test_basic(float *d_data, int iterations) {
    nvtxMark("test_basic: start");

    for (int iter = 0; iter < iterations; iter++) {
        nvtxRangePush("Region A: Vector Add");

        // 分配内存
        float *d_a, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

        // 初始化
        CUDA_CHECK(cudaMemset(d_a, 1, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b, 2, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

        nvtxMark("Region A: kernel launch");
        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add<<<blocks, BLOCK_SIZE>>>(d_out, d_a, d_b, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        nvtxMark("Region A: memcpy H2D");
        CUDA_CHECK(cudaMemcpy(d_data, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        // 清理
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_out));

        nvtxRangePop();
    }

    {
        nvtxRangePush("Region B: Vector Mul");
        float *d_a, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

        CUDA_CHECK(cudaMemset(d_a, 3, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b, 4, N * sizeof(float)));

        nvtxMark("Region B: kernel launch");
        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_mul<<<blocks, BLOCK_SIZE>>>(d_out, d_a, d_b, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_out));

        nvtxRangePop();
    }

    nvtxMark("test_basic: end");
}

// ============================================================================
// 自定义域测试 (C API)
// ============================================================================
/* ---------------- domain test ---------------- */
void test_domain(float *d_data, int iterations) {
    nvtxDomainHandle_t domainA = nvtxDomainCreateA("Domain A");

    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size    = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;

    attr.message.ascii = "test_domain: start";
    nvtxDomainMarkEx(domainA, &attr);

    for (int iter = 0; iter < iterations; iter++) {

        attr.message.ascii = "Domain A@Iteration";
        nvtxDomainRangePushEx(domainA, &attr);

        float *d_a, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

        attr.message.ascii = "Domain A@Kernel";
        nvtxDomainRangePushEx(domainA, &attr);

        CUDA_CHECK(cudaMemset(d_a, 1, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b, 2, N * sizeof(float)));

        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add<<<blocks, BLOCK_SIZE>>>(d_out, d_a, d_b, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        nvtxDomainRangePop(domainA);  // Kernel

        CUDA_CHECK(cudaMemcpy(d_data, d_out,
                               N * sizeof(float),
                               cudaMemcpyDeviceToHost));

        nvtxDomainRangePop(domainA);  // Iteration

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_out));
    }

    attr.message.ascii = "test_domain: end";
    nvtxDomainMarkEx(domainA, &attr);

    nvtxDomainDestroy(domainA);
}

// ============================================================================
// 嵌套范围测试
// ============================================================================
void test_nested(float *d_data, int iterations) {
    nvtxMark("test_nested: start");

    for (int iter = 0; iter < iterations; iter++) {
        nvtxRangePush("Level 1: Outer Loop");

        {
            nvtxRangePush("Level 2: Vector Add Setup");

            float *d_a, *d_b, *d_out;
            CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

            nvtxMark("Level 2: Memory allocated");

            {
                nvtxRangePush("Level 3: Kernel Launch");
                CUDA_CHECK(cudaMemset(d_a, 1, N * sizeof(float)));
                CUDA_CHECK(cudaMemset(d_b, 2, N * sizeof(float)));
                CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

                int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
                vector_add<<<blocks, BLOCK_SIZE>>>(d_out, d_a, d_b, N);
                CUDA_CHECK(cudaDeviceSynchronize());

                {
                    nvtxRangePush("Level 4: Post-Kernel");
                    nvtxMark("Level 4: Kernel completed");
                    CUDA_CHECK(cudaMemcpy(d_data, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
                    nvtxRangePop();
                }

                CUDA_CHECK(cudaFree(d_a));
                CUDA_CHECK(cudaFree(d_b));
                CUDA_CHECK(cudaFree(d_out));

                nvtxRangePop();  // Level 3
            }
            nvtxRangePop();  // Level 2
        }

        // 测试更深的嵌套
        {
            nvtxRangePush("Deep Level 1");
            nvtxRangePush("Deep Level 2");
            nvtxRangePush("Deep Level 3");
            nvtxRangePush("Deep Level 4");
            nvtxRangePush("Deep Level 5");
            nvtxMark("Deep nesting complete (5 levels)");
            nvtxRangePop();  // Level 5
            nvtxRangePop();  // Level 4
            nvtxRangePop();  // Level 3
            nvtxRangePop();  // Level 2
            nvtxRangePop();  // Level 1
        }
    }

    nvtxMark("test_nested: end");
}

// ============================================================================
// 标记点详细测试
// ============================================================================
void test_mark(float *d_data, int iterations) {
    nvtxMark("test_mark: start");

    // 1. 简单标记
    nvtxMark("Simple mark: before loop");

    for (int iter = 0; iter < iterations; iter++) {
        nvtxMark("Mark: iteration start");

        float *d_a, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

        // 2. 序列标记
        nvtxMark("Mark: allocate memory");
        CUDA_CHECK(cudaMemset(d_a, 1, N * sizeof(float)));

        nvtxMark("Mark: memset d_a done");
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
        nvtxMark("Mark: memset d_out done");

        // 3. 关键点标记
        nvtxMark("Mark: before kernel");
        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add<<<blocks, BLOCK_SIZE>>>(d_out, d_a, g_d_b, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxMark("Mark: after kernel");

        nvtxMark("Mark: before memcpy");
        CUDA_CHECK(cudaMemcpy(d_data, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        nvtxMark("Mark: after memcpy");

        nvtxMark("Mark: before free");
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_out));
        nvtxMark("Mark: after free");

        nvtxMark("Mark: iteration end");
    }

    // 4. 总结标记
    nvtxMark("Mark: all iterations complete");
    nvtxMark("test_mark: end");
}

// ============================================================================
// 函数级自动范围测试 (NVTX3_FUNC_RANGE宏)
// ============================================================================
void function_with_auto_range_1() {
    NVTX3_FUNC_RANGE();  // 自动命名为 "function_with_auto_range_1"
    nvtxMark("Inside function_with_auto_range_1");
}

void function_with_auto_range_2() {
    NVTX3_FUNC_RANGE();  // 自动命名为 "function_with_auto_range_2"
    nvtxMark("Inside function_with_auto_range_2");
}

void function_with_auto_range_3() {
    NVTX3_FUNC_RANGE();  // 自动命名为 "function_with_auto_range_3"
    nvtxMark("Inside function_with_auto_range_3");
}

void test_func_auto(float *d_data, int iterations) {
    NVTX3_FUNC_RANGE();  // 主函数自动范围

    nvtxMark("test_func_auto: start");

    for (int iter = 0; iter < iterations; iter++) {
        nvtxMark("Iteration start");

        // 调用带自动范围的函数
        function_with_auto_range_1();
        function_with_auto_range_2();
        function_with_auto_range_3();

        // 执行一些CUDA操作
        float *d_a, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

        CUDA_CHECK(cudaMemset(d_a, 1, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add<<<blocks, BLOCK_SIZE>>>(d_out, d_a, g_d_b, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(d_data, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_out));
    }

    nvtxMark("test_func_auto: end");
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    const char* mode = "basic";
    int iterations = 5;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        } else if (strncmp(argv[i], "--iter=", 7) == 0) {
            iterations = atoi(argv[i] + 7);
        }
    }

    printf("NVTX Demo - Mode: %s, Iterations: %d\n", mode, iterations);

    // 分配主机内存
    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));

    // 初始化全局变量（用于test_mark）
    CUDA_CHECK(cudaMalloc(&g_d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_d_b, 2, N * sizeof(float)));

    // 根据模式运行测试
    if (strcmp(mode, "basic") == 0) {
        test_basic(h_data, iterations);
    } else if (strcmp(mode, "domain") == 0) {
        test_domain(h_data, iterations);
    } else if (strcmp(mode, "nested") == 0) {
        test_nested(h_data, iterations);
    } else if (strcmp(mode, "mark") == 0) {
        test_mark(h_data, iterations);
    } else if (strcmp(mode, "func-auto") == 0) {
        test_func_auto(h_data, iterations);
    } else if (strcmp(mode, "all") == 0) {
        printf("=== Running all tests sequentially ===\n\n");
        printf("--- Basic test ---\n");
        test_basic(h_data, iterations);
        printf("\n--- Domain test ---\n");
        test_domain(h_data, iterations);
        printf("\n--- Nested test ---\n");
        test_nested(h_data, iterations);
        printf("\n--- Mark test ---\n");
        test_mark(h_data, iterations);
        printf("\n--- Func-auto test ---\n");
        test_func_auto(h_data, iterations);
    } else {
        printf("Unknown mode: %s\n", mode);
        print_help(argv[0]);
        return 1;
    }

    // 清理
    CUDA_CHECK(cudaFree(g_d_b));
    CUDA_CHECK(cudaFreeHost(h_data));

    printf("\nNVTX demo completed.\n");
    return 0;
}
