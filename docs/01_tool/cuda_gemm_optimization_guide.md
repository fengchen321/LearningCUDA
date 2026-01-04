# CUDA GEMM 优化指南

> 本指南为初学者提供矩阵乘法（GEMM）CUDA 优化的系统性方法论，后续所有内核优化均应遵循此框架。

## 目录

- [优化层次概览](#优化层次概览)
- [优化前准备](#优化前准备)
- [第一阶段：基础正确性](#第一阶段基础正确性)
- [第二阶段：内存访问优化](#第二阶段内存访问优化)
- [第三阶段：计算效率优化](#第三阶段计算效率优化)
- [第四阶段：高级优化](#第四阶段高级优化)
- [优化检查清单](#优化检查清单)
- [内核命名规范](#内核命名规范)
- [性能对比模板](#性能对比模板)

---

## 优化层次概览

CUDA 优化可以从多个层次进行理解和实施。理解这些层次有助于系统性地进行优化，而不是盲目尝试。

```text
┌─────────────────────────────────────────────────────────────────┐
│                     第四阶段：高级优化                           │
│         Tensor Core / 混合精度 / 多流并行 / 算法创新             │
├─────────────────────────────────────────────────────────────────┤
│                     第三阶段：计算效率                           │
│         指令优化 / 寄存器优化 / 数学库优化 / 分支预测             │
├─────────────────────────────────────────────────────────────────┤
│                     第二阶段：内存访问                           │
│         共享内存 / 缓存利用 / 内存合并 / 带宽优化                 │
├─────────────────────────────────────────────────────────────────┤
│                     第一阶段：基础正确性                         │
│         正确性验证 / 线程配置 / 基本并行化                       │
└─────────────────────────────────────────────────────────────────┘
```

**优化顺序建议**：必须按顺序进行，先确保正确性，再追求性能。低层优化是高层优化的基础。

---

## 优化前准备

### 1.1 环境验证

在开始优化之前，确认开发环境正常工作：

```bash
# 1. 检查 CUDA 编译器
nvcc --version

# 2. 检查 GPU 驱动
nvidia-smi

# 3. 运行示例项目确认编译正常
cd build && cmake .. && cmake --build .
./GEMM/profile_cuda_gemm_fp32
```

### 1.2 建立基准性能数据

优化必须有数据支撑。在修改任何代码之前，先收集基准数据：

```bash
# 进入构建目录
cd build

# 运行程序获取基准时间
./GEMM/profile_cuda_gemm_fp32

# 使用 Nsight Systems 进行初步分析
nsys profile --gpu-metrics-device=0 --output=baseline_nsys ./GEMM/profile_cuda_gemm_fp32

# 使用 Nsight Compute 获取详细指标
ncu --metrics collection=SpeedOfLight --kernel-name="gemm_kernel_v00*" --output=baseline_ncu ./GEMM/profile_cuda_gemm_fp32
```

### 1.3 性能指标记录模板

创建 `performance_log.md` 记录每次优化的性能变化：

```markdown
# 性能记录

## 基准数据（v00 Naive）
- 矩阵大小：1024x1024
- 执行时间：XX.XX ms
- SM 利用率：XX%
- 内存带宽：XX GB/s
- 测试日期：2024-XX-XX

## 优化记录

### v01: 共享内存
- 修改内容：添加分块和共享内存缓存
- 性能提升：XX%
- SM 利用率：XX%
- 备注：...
```

---

## 第一阶段：基础正确性

### 1.1 线程配置基础

线程配置是并行计算的基础。理解 Grid 和 Block 的维度配置：

```cpp
// 基础配置原则
dim3 const block_dim{16U, 16U, 1U};  // 256 threads per block
dim3 const grid_dim{
    (static_cast<unsigned int>(B.width) + block_dim.x - 1U) / block_dim.x,
    (static_cast<unsigned int>(A.height) + block_dim.y - 1U) / block_dim.y, 1U};
```

**关键参数说明**：
- `blockDim.x/y`: 每个 block 的线程数，通常选择 16x16=256 或 32x8=256
- `gridDim.x/y`: 需要覆盖整个输出矩阵所需的 block 数量
- 每个线程计算一个输出元素（最基础的并行化策略）

### 1.2 正确性验证

实现任何优化之前，必须确保结果正确：

```cpp
// 在 CPU 端进行验证的代码示例
bool verify_result(const float* A, const float* B, const float* C,
                   int M, int N, int K) {
    const float eps = 1e-3f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += A[i * K + k] * B[k * N + j];
            }
            if (fabs(C[i * N + j] - expected) > eps) {
                return false;
            }
        }
    }
    return true;
}
```

### 1.3 第一阶段检查清单

```text
□ 1.1 线程索引计算正确
   - 验证 threadIdx.x/y, blockIdx.x/y 的使用
   - 确认没有线程越界访问

□ 1.2 边界条件处理
   - 对于非整除的矩阵大小，添加边界检查
   - 越界时返回 0 或跳过

□ 1.3 结果验证
   - 与 CPU 参考实现对比
   - 使用小矩阵进行完整验证

□ 1.4 错误处理
   - 添加 CUDA 错误检查 CHECK_LAST_CUDA_ERROR()
   - 检查内存分配返回值

□ 1.5 基准性能记录
   - 记录优化前的执行时间
   - 记录 GPU 利用率
```

---

## 第二阶段：内存访问优化

### 2.1 内存访问模式分析

理解 GPU 内存层次结构是优化的关键：

```text
┌─────────────────────────────────────────────────────────────────┐
│  内存类型         │  带宽（相对）  │  延迟（相对）  │  作用域    │
├─────────────────────────────────────────────────────────────────┤
│  寄存器           │  最高          │  最低          │  线程      │
│  共享内存         │  高            │  低            │  线程块    │
│  L1/L2 缓存       │  中            │  中            │  多 block  │
│  全局内存         │  低            │  高            │  全局      │
└─────────────────────────────────────────────────────────────────┘
```

**GEMM 的内存访问特点**：
- 全局内存访问是主要瓶颈
- 每次计算需要读取 A 的一行和 B 的一列
- 数据重用机会：A 的行元素被使用 N 次，B 的列元素被使用 M 次

### 2.2 分块策略（Tiling）

分块是 GEMM 优化的核心技术。通过将矩阵划分为小块，可以将数据缓存到共享内存中。

**v01 实现原理**：

```cpp
// 核心思想：将计算划分为小块，每个线程块计算 C 的一个 BLOCK_SIZE x BLOCK_SIZE 子块
// 通过共享内存缓存 A 和 B 的子块，避免重复的全局内存访问

__shared__ T As[BLOCK_SIZE][BLOCK_SIZE];  // 缓存 A 的子块
__shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];  // 缓存 B 的子块

for (int m = 0; m < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
    // 1. 将 A 和 B 的子块加载到共享内存
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);

    // 2. 同步确保数据就绪
    __syncthreads();

    // 3. 使用共享内存中的数据进行计算
    for (int e = 0; e < BLOCK_SIZE; ++e) {
        Cvalue += As[row][e] * Bs[e][col];
    }

    // 4. 同步，准备下一轮迭代
    __syncthreads();
}
```

### 2.3 内存合并（Coalescing）

确保相邻线程访问相邻的内存地址：

```cpp
// 良好的合并访问模式
// 线程 col 访问 B[e * B.width + col]，相邻线程访问相邻地址 ✓

// 优化前（可能不合并）
// Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];

// 优化后（确保合并访问）
// 确保 A 按行存储，B 按列存储时考虑访问模式
```

### 2.4 第二阶段检查清单

```text
□ 2.1 分块大小选择
   - BLOCK_SIZE 通常为 16 或 32
   - 确保共享内存足够容纳两个 BLOCK_SIZE x BLOCK_SIZE 块
   - 考虑寄存器压力和占用率的平衡

□ 2.2 共享内存使用
   - 正确声明 __shared__ 内存
   - 使用 __syncthreads() 同步
   - 避免共享内存 bank 冲突

□ 2.3 内存合并
   - 相邻线程访问相邻内存地址
   - 验证 A 和 B 的访问模式

□ 2.4 性能提升验证
   - 使用 ncu 检查 dram__bytes.sum 是否减少
   - 对比优化前后的执行时间
```

### 2.5 内存优化代码模板

```cpp
// 第二阶段优化代码模板

#define BLOCK_SIZE 16  // 可调整为 32 进行测试

template <typename T>
__global__ void gemm_vXX(const Matrix<T> A, const Matrix<T> B, Matrix<T> C)
{
    // 1. 获取线程在 block 中的位置
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    // 2. 声明共享内存
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 3. 计算当前线程需要处理的子矩阵
    Matrix<T> Csub = GetSubMatrix(C, blockRow, blockCol);
    T Cvalue = 0;

    // 4. 遍历所有子矩阵块
    for (int m = 0; m < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        // 5. 加载数据到共享内存（合并访问）
        Matrix<T> Asub = GetSubMatrix(A, blockRow, m);
        Matrix<T> Bsub = GetSubMatrix(B, m, blockCol);

        // 线程加载一个元素
        if (row < Asub.height && col < Asub.width) {
            As[row][col] = GetElement(Asub, row, col);
        } else {
            As[row][col] = 0;
        }

        if (row < Bsub.height && col < Bsub.width) {
            Bs[row][col] = GetElement(Bsub, row, col);
        } else {
            Bs[row][col] = 0;
        }

        // 6. 同步
        __syncthreads();

        // 7. 计算
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        // 8. 同步，准备下一轮
        __syncthreads();
    }

    // 9. 写回结果
    if (row < Csub.height && col < Csub.width) {
        SetElement(Csub, row, col, Cvalue);
    }
}
```

---

## 第三阶段：计算效率优化

### 3.1 占用率优化

占用率是指每个 SM 上活跃线程束的比例。较高的占用率有助于隐藏内存延迟。

```bash
# 使用 Nsight Compute 检查占用率
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained \
    --kernel-name="gemm_kernel_v01*" ./program

# 查看限制占用率的因素
ncu --metrics \
    sm__occupancy.block_per_sm.max, \
    sm__occupancy.threads.max.per_sm, \
    sm__regfile_hybrid_peak_operand_per_warp.pct, \
    sm__shared_peak_operand_per_warp.pct \
    --kernel-name="gemm_kernel_v01*" ./program
```

### 3.2 寄存器优化

每个线程使用的寄存器数量会影响占用率。使用 `-maxrregcount` 或 `__launch_bounds__` 限制：

```cpp
// 方法 1：使用 __launch_bounds__ 提示编译器
__global__ void __launch_bounds__(256, 2)  // max threads per block, min blocks per SM
gemm_vXX(const Matrix<T> A, const Matrix<T> B, Matrix<T> C)
{
    // ...
}

// 方法 2：在编译时指定
// cmake 中添加：-Xmaxrregcount=32
```

### 3.3 循环展开

对于小的循环，编译器可能自动展开。也可以手动展开：

```cpp
// 编译器可能自动展开这个循环
for (int e = 0; e < BLOCK_SIZE; ++e) {
    Cvalue += As[row][e] * Bs[e][col];
}

// 手动展开（如果编译器不自动处理）
#pragma unroll
for (int e = 0; e < BLOCK_SIZE; ++e) {
    Cvalue += As[row][e] * Bs[e][col];
}
```

### 3.4 分支优化

避免线程束内的分支分化：

```cpp
// 边界检查可能导致分支分化
if (row < Asub.height && col < Asub.width) {  // 所有线程可能走不同分支
    As[row][col] = GetElement(Asub, row, col);
} else {
    As[row][col] = 0;
}

// 优化方案：使用三目运算符（分支预测可能更高效）
As[row][col] = (row < Asub.height && col < Asub.width) ?
               GetElement(Asub, row, col) : 0;
```

### 3.5 第三阶段检查清单

```text
□ 3.1 占用率检查
   - SM 利用率是否 > 80%？
   - 限制因素是寄存器还是共享内存？

□ 3.2 寄存器使用
   - 使用了多少寄存器？
   - 是否可以通过降低占用率换取更好的指令级并行？

□ 3.3 循环效率
   - 编译器是否正确展开循环？
   - 是否需要手动添加 #pragma unroll

□ 3.4 分支效率
   - 是否存在线程束分化？
   - 边界检查是否可以优化？

□ 3.5 数学运算优化
   - 是否可以使用更高效的操作？
   - 重复计算是否可以避免？
```

---

## 第四阶段：高级优化

### 4.1 增加分块大小

更大的 BLOCK_SIZE 可以提高数据重用率，但会降低占用率：

```cpp
// 尝试不同的分块大小
#define BLOCK_SIZE 32  // 32x32 = 1024 threads per block
#define BLOCK_SIZE 64  // 更大分块，需要更多共享内存
```

### 4.2 双缓冲（Double Buffering）

在计算当前块数据时，预取下一块数据：

```cpp
// 伪代码示例
__shared__ T As[2][BLOCK_SIZE][BLOCK_SIZE];
__shared__ T Bs[2][BLOCK_SIZE][BLOCK_SIZE];

int writeIdx = 0;
for (int m = 0; m < numBlocks; ++m) {
    int readIdx = writeIdx;

    // 异步预取下一块（需要 CUDA 7.0+）
    load_async(As[writeIdx], Asub_next);
    load_async(Bs[writeIdx], Bsub_next);

    __syncthreads();

    // 使用上一块数据计算
    compute(As[readIdx], Bs[readIdx]);

    writeIdx = 1 - writeIdx;
}
```

### 4.3 流水线优化

使用 CUDA 7.0+ 的异步内存复制指令：

```cpp
// 使用 cudaMemcpyAsync 和共享内存
// 详细实现见 NVIDIA CUDA 示例代码
```

### 4.4 混合精度（FP16/BF16）

使用半精度浮点数可以提高吞吐量和带宽利用率：

```cpp
// 使用 __half 或 nv_bfloat16
#include <cuda_fp16.h>

__global__ void gemm_fp16(const Matrix<__half> A,
                          const Matrix<__half> B,
                          Matrix<__half> C) {
    // 使用 __half2 指令同时处理两个值
}
```

### 4.5 Tensor Core 加速

使用 WMMA（Warpt Matrix Multiply Accumulate）API：

```cpp
#include <mma.h>

using namespace nvcuda::wmma;

__global__ void gemm_tensor(const Matrix<float> A,
                            const Matrix<float> B,
                            Matrix<float> C) {
    fragment<matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half> a_frag;
    fragment<matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half> b_frag;
    fragment<accumulator, BLOCK_M, BLOCK_N, BLOCK_K, float> c_frag;

    // 加载数据
    load_matrix_sync(a_frag, A_data, BLOCK_K);
    load_matrix_sync(b_frag, B_data, BLOCK_K);

    // 执行矩阵乘法
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 存储结果
    store_matrix_sync(C_data, c_frag, BLOCK_N);
}
```

### 4.6 第四阶段检查清单

```text
□ 4.1 分块大小测试
   - 尝试 16, 32, 64 等不同大小
   - 找到占用率和数据重用的最佳平衡点

□ 4.2 异步操作
   - 是否可以使用 cudaMemcpyAsync？
   - 共享内存加载和计算是否可以重叠？

□ 4.3 精度选择
   - 是否需要 FP32 精度？
   - FP16/BF16 是否足够？

□ 4.4 Tensor Core
   - GPU 是否支持 Tensor Core？
   - 是否使用 WMMA API？

□ 4.5 多流并行
   - 是否可以流水线处理多个矩阵？
   - 是否可以使用 CUDA 流？
```

---

## 优化检查清单

### 综合检查清单

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基础正确性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 线程索引计算正确，无越界访问
□ 边界条件正确处理
□ 与 CPU 参考实现结果一致
□ 添加 CUDA 错误检查
□ 建立基准性能数据

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
内存优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 实现分块策略（tiling）
□ 正确使用共享内存
□ 添加必要的 __syncthreads()
□ 内存访问合并（coalesced access）
□ 减少全局内存访问次数
□ 避免共享内存 bank 冲突

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
计算优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 占用率在合理范围（>50%）
□ 寄存器使用合理
□ 循环效率优化（展开）
□ 分支预测优化
□ 减少冗余计算

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
高级优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 测试不同分块大小
□ 考虑双缓冲/流水线
□ 评估混合精度可行性
□ 考虑 Tensor Core（如果支持）
□ 使用性能分析工具验证优化效果
```

---

## 内核命名规范

为了保持项目的可维护性，所有内核应遵循以下命名规范：

### 版本号规则

```text
v00  - 基础版本（Naive，无优化）
v01  - 共享内存优化（分块策略）
v02  - 增加分块大小 / 双缓冲
v03  - 寄存器优化 / 循环展开
v04  - 混合精度（FP16）
v05  - Tensor Core / WMMA
vXX  - 其他特定优化
```

### 文件命名

```text
src/00_Naive_MatMul.cu      -> gemm_v00
src/01_Share_MatMul.cu      -> gemm_v01
src/02_XX_Opt_MatMul.cu     -> gemm_v02
...
```

### 头文件声明

```cpp
// cuda_gemm.hpp
template <typename T>
void launch_gemm_kernel_v00(const Matrix<T> A, const Matrix<T> B, Matrix<T> C, cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v01(const Matrix<T> A, const Matrix<T> B, Matrix<T> C, cudaStream_t stream);

// 新增内核按顺序添加
template <typename T>
void launch_gemm_kernel_v02(const Matrix<T> A, const Matrix<T> B, Matrix<T> C, cudaStream_t stream);
```

### 添加新内核的步骤

1. 创建新文件 `src/XX_NewFeature_MatMul.cu`
2. 实现对应的 `gemm_vXX` 内核和 `launch_gemm_kernel_vXX` 函数
3. 在 `cuda_gemm.hpp` 中添加函数声明
4. 在 `CMakeLists.txt` 中添加源文件
5. 在 `profile_cuda_gemm_fp32.cu` 中注册新的内核
6. 更新本文档，记录新版本的优化点

---

## 性能对比模板

### 性能对比表

| 版本 | 描述 | 执行时间 (ms) | SM 利用率 | 内存带宽 (GB/s) | 相对性能 |
|------|------|---------------|-----------|-----------------|----------|
| v00 | Naive 基础实现 | - | - | - | 1.0x |
| v01 | 共享内存 + 分块 | - | - | - | ?.x |
| v02 | ... | - | - | - | ?.x |

### 性能测试脚本

```bash
#!/bin/bash
# benchmark.sh - 性能对比脚本

EXECUTABLE="./GEMM/profile_cuda_gemm_fp32"
OUTPUT="benchmark_results.txt"

echo "GEMM 性能基准测试" > $OUTPUT
echo "测试时间: $(date)" >> $OUTPUT
echo "=================================" >> $OUTPUT

for run in {1..5}; do
    echo "运行 $run..."
    echo "--- 运行 $run ---" >> $OUTPUT
    $EXECUTABLE 2>&1 | tee -a $OUTPUT
done

echo "=================================" >> $OUTPUT
echo "测试完成"
```

### Nsight Compute 对比脚本

```bash
#!/bin/bash
# compare_kernels.sh - 内核对比脚本

KERNELS=("v00" "v01" "v02")

for kernel in "${KERNELS[@]}"; do
    echo "分析 kernel: $kernel"
    ncu --metrics collection=SpeedOfLight \
        --kernel-name="gemm_kernel_${kernel}*" \
        --export=${kernel}_metrics.json \
        ./GEMM/profile_cuda_gemm_fp32
done

echo "分析完成，结果保存在 *_metrics.json"
```

---

## 参考资源

### 官方文档

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### 推荐阅读

- [CUDA Samples - matrixMul](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/matrixMul)
- [Volta Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
- [Tensor Core API Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma)

### 性能分析工具

```bash
# Nsight Systems 入门
nsys profile ./program
nsight-systems report.nsys-rep

# Nsight Compute 入门
ncu --metrics collection=SpeedOfLight ./program
ncu-ui report.ncu-rep
```

---

## 附录：常见问题

### Q1: 优化后性能反而下降？

可能原因：
- 占用率过低
- 共享内存 bank 冲突
- 编译器优化被意外禁用

使用 Nsight Compute 检查具体原因。

### Q2: 如何确定最优的分块大小？

建议：
- 从 16 开始测试
- 逐步增加到 32, 64
- 观察占用率和执行时间的变化
- 找到最佳平衡点

### Q3: 共享内存不够用怎么办？

解决方案：
- 减小分块大小
- 使用双缓冲（每次只加载一部分）
- 考虑使用 L1 缓存替代部分共享内存

### Q4: 如何验证优化效果？

验证步骤：
1. 确保结果正确
2. 多次运行取平均值
3. 使用 Nsight 工具收集详细指标
4. 对比优化前后的各项指标
