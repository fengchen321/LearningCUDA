# profile_demo - Nsight Systems 性能分析测试 Demo 集合

## 目录结构

```
profile_demo/
├── CMakeLists.txt
├── README.md
├── 01_basic/              # 基础向量加法
├── 02_memory/            # 内存传输测试
├── 03_multistream/       # 多流测试
├── 04_nvtx/              # NVTX标记测试
├── 05_unified_memory/    # 统一内存测试
├── 06_cuda_graph/        # CUDA Graphs测试
├── 07_callstack/         # 调用链测试
├── 08_fork_exec/         # fork-exec测试
├── 09_mpi/               # MPI多进程测试
└── include/              # 公共头文件
```

## 快速开始

```bash
# 编译所有 demo
cd profile_demo
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j$(nproc)

# 运行基础测试
cd ../01_basic
./basic

# 使用 nsys 采集
nsys profile --stats=true -o basic_report ./basic

# 使用 nsys-ui 查看结果
nsys-ui basic_report.nsys-rep
```

## Demo 列表与 nsys 测试场景

| Demo | 文件 | 用途 | nsys 测试参数 |
|------|------|------|--------------|
| 01_basic | basic.cu | 简单向量加法 | 基础采集 |
| 02_memory | memory_test.cu | 大数组H2D/D2H传输 | 内存传输分析 |
| 03_multistream | multistream.cu | 异步多流执行 | 多流追踪 |
| 04_nvtx | nvtx_demo.cu | 自定义NVTX标记 | `--nvtx`, `--nvtx-include/exclude` |
| 05_unified_memory | um_demo.cu | 统一内存访问 | `--cuda-um-*-page-faults` |
| 06_cuda_graph | cuda_graph.cu | CUDA Graphs | `--cuda-graph-trace` |
| 07_callstack | callstack.cu | 嵌套核函数调用 | `--sample`, `--backtrace` |
| 08_fork_exec | fork_exec.cu | fork后exec | `--trace-fork-before-exec` |
| 09_mpi | mpi_demo.cu | MPI多进程通信 | `--mpi-impl`, `--trace=mpi` |

## nsys 常用测试命令

### 基础采集
```bash
nsys profile --stats=true -o report_name ./demo
```

### 内存传输分析
```bash
nsys profile --stats=true -t cuda,osrt -o memory_trace ./memory_test
```

### NVTX 范围过滤
```bash
# 只采集指定范围
nsys profile --nvtx --nvtx-include="Computation@Region A" -o nvtx_report ./nvtx_demo

# 排除指定范围
nsys profile --nvtx --nvtx-exclude="Region B" -o nvtx_exclude ./nvtx_demo
```

### 统一内存诊断
```bash
nsys profile --stats=true --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true -o um_report ./um_demo
```

### CUDA Graphs 分析
```bash
nsys profile --stats=true --cuda-graph-trace=graph -o graph_report ./cuda_graph
```

### CPU 回溯采样
```bash
# LBR 回溯 (快速)
nsys profile --sample=process-tree --backtrace=lbr -o lbr_report ./callstack

# DWARF 回溯 (完整)
nsys profile --sample=process-tree --backtrace=dwarf -o dwarf_report ./callstack
```

### MPI 多进程
```bash
# OpenMPI
nsys profile --stats=true --mpi=openmpi -o mpi_report mpirun -np 4 ./mpi_demo

# MPICH
nsys profile --mpi=mpich -o mpi_mpich mpirun -np 4 ./mpi_demo
```

### 系统级分析 (需要 root)
```bash
sudo nsys profile --stats=true --sample=system-wide \
    --ftrace=drm/* --event-sample=system-wide -o system_report ./basic
```

### GPU 指标采集
```bash
nsys profile --stats=true --gpu-metrics-devices=all \
    --gpu-metrics-frequency=20000 -o gpu_metrics ./basic
```

## 依赖

- CUDA Toolkit 11.0+
- CMake 3.18+
- MPI (仅 09_mpi demo)
- NVTX3 (仅 04_nvtx demo)
