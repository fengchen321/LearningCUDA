# GEMM Kernel 性能分析

## 背景

在 NVIDIA GeForce RTX 3060 Ti (WSL2) 上测试时，发现 V01 (Shared Memory) 版本性能反而不如 V00 (Naive)：

```
V00: Effective Bandwidth: 0.50 GB/s, 性能: cuBLAS 的 0.69%
V01: Effective Bandwidth: 0.28 GB/s, 性能: cuBLAS 的 0.36%
```

在数据中心卡（如 L20/A100）上 V01 应该更优，此异常需要分析。

## 问题分析

### V00 (Naive) 实现特点
- 每个线程计算 C 的一个输出元素
- 直接从 global memory 读取 A 和 B 元素
- 无额外同步开销

### V01 (Shared Memory) 实现特点
```cuda
#define BLOCK_SIZE 16

for (int m = 0; m < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
    // 加载 tile 到 shared memory
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 每个循环同步 2 次
    __syncthreads();  // after load

    for (int e = 0; e < BLOCK_SIZE; ++e) {
        Cvalue += As[row][e] * Bs[e][col];
    }

    __syncthreads();  // before next load
}
```

### 性能问题根源

| 问题 | 说明 |
|------|------|
| **Tile 数量过多** | K=512 需要 32 个 tile (512/16)，每个线程循环 32 次 |
| **同步开销过大** | 每个循环 2 次 `__syncthreads()`，共 64 次同步 |
| **Tile 粒度过小** | 16x16 的 tile 太小，无法有效隐藏内存延迟 |
| **未充分利用 shared memory** | 每次只处理一个 tile，没有累积计算 |

### 为什么在高端卡上表现更好？

| 因素 | RTX 3060 Ti (WSL2) | L20/A100 |
|------|-------------------|----------|
| 内存带宽 | ~448 GB/s | ~2-3 TB/s |
| SM 数量 | 38 | 80+ |
| WSL2 开销 | 有额外开销 | 无 |
| 同步占比 | 高 | 低（被高带宽隐藏） |

在高端卡上，高带宽可以隐藏同步开销；在低端卡+WSL2 上，同步开销占比过高导致性能下降。

## 验证方法

### 1. 使用 nvprof 分析

```bash
nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./GEMM/profile_cuda_gemm_fp32
```

**关键指标**：
- `achieved_occupancy`: 实际occupancy
- `gld_efficiency`: 全局内存加载效率
- `gst_efficiency`: 全局内存存储效率

### 2. 使用 Nsight Compute (推荐)

```bash
# 生成 V00 报告
ncu --metrics sm__throughput.avg.pct_of_peak_sustained, \
             sm__warps_active.avg.pct_of_peak_sustained, \
             dram__bytes.sum, \
             sm__warps_launched.sum \
    --target-processes all \
    -o profile_v00 ./GEMM/profile_cuda_gemm_fp32

# 生成 V01 报告
ncu --metrics sm__throughput.avg.pct_of_peak_sustained, \
             sm__warps_active.avg.pct_of_peak_sustained, \
             dram__bytes.sum, \
             sm__warps_launched.sum \
    --target-processes all \
    -o profile_v01 ./GEMM/profile_cuda_gemm_fp32
```

**对比指标**：

| 指标 | 含义 | V00 预期 | V01 预期 |
|------|------|----------|----------|
| `sm__throughput.avg.pct_of_peak_sustained` | SM 利用率 | 低（内存瓶颈） | 稍高 |
| `sm__warps_active.avg.pct_of_peak_sustained` | Warp 活跃率 | 低 | 可能更低 |
| `dram__bytes.sum` | 全局内存访问量 | 高 | **应该更低** |
| `sm__warps_launched.sum` | 发射 warp 数 | 多 | 少 |

**验证预期**：
- V01 的 `dram__bytes` 应该明显更低（验证 shared memory 生效）
- V01 的 `sm__throughput` 应该更低（验证同步开销过大）

### 3. 代码中添加计时

在 `01_Share_MatMul.cu` 的 kernel 中添加 CUDA event 计时来测量同步开销：

```cuda
__global__ void gemm_v01_profiled(...) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录同步开始
    cudaEventRecord(start);
    __syncthreads();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sync_time;
    cudaEventElapsedTime(&sync_time, start, stop);

    // ... kernel 逻辑 ...
}
```

### 4. 使用 CUDA Profiler

```bash
# 设置环境变量
export CUDA_PROFILE=1
export CUDA_PROFILE_LOG=profile.log

./GEMM/profile_cuda_gemm_fp32

# 查看 log
cat profile.log
```

## 优化方向

### 短期优化 (改动小)

1. **增大 BLOCK_SIZE**：16 → 32 或 64
2. **减少 tile 数量**：K=512 时，BLOCK_SIZE=32 则只有 16 个 tile

### 长期优化 (参考 cuBLAS)

1. **Pipelining**：在计算当前 tile 时预加载下一个 tile
2. **Register tiling**：将数据保持在 registers 中而非反复访问 shared memory
3. **Bank conflict optimization**：使用 padding 避免 shared memory bank conflict
4. **Warp tiling**：使用 warp-level 操作

## 参考资料

- [CUDA C Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Matrix Multiplication Optimization](https://siboehm.com/2021/11/20/gemm-optimization/)
