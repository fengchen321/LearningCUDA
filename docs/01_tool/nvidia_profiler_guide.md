# NVIDIA 性能分析工具使用指南

本指南详细介绍如何使用 NVIDIA Nsight Systems 和 Nsight Compute 对 CUDA 程序进行性能分析。这两个工具是 CUDA 开发中最常用的性能分析工具，能够帮助开发者深入了解程序的运行行为，找出性能瓶颈并进行优化。

## 1 环境准备

### 1.1 检查工具安装

在使用性能分析工具之前，首先需要确认这些工具已经正确安装。以下是检查工具是否可用的命令：

```bash
# 检查 Nsight Systems 是否安装
nsys --version

# 检查 Nsight Compute 是否安装
ncu --version

# 检查 CUDA 工具链版本
nvcc --version

# 查看当前 GPU 信息
nvidia-smi
```

如果上述命令能够正常输出版本信息，说明工具已经正确安装。如果某个工具未安装，需要通过 NVIDIA 开发者网站下载并安装 CUDA Toolkit，Nsight Systems 和 Nsight Compute 通常随 CUDA Toolkit 一起分发。

### 1.2 编译程序

在进行性能分析之前，需要确保程序已经使用调试信息编译。调试信息能够帮助性能分析工具提供更详细的函数名、内核名称和源代码行号信息。以下是在 CMake 项目中启用调试信息的编译命令：

```bash
# 创建构建目录
mkdir -p build && cd build

# 使用调试信息重新配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 或者显式指定 CUDA 调试标志
cmake .. -DCMAKE_CUDA_FLAGS="-g -G"

# 编译程序
cmake --build . --config Debug
```

`-g` 选项添加主机代码的调试信息，`-G` 选项添加设备代码（CUDA 内核）的调试信息。对于较旧的 CUDA 版本，可能需要使用 `--debug` 标志来启用设备代码调试。对于 Maxwell 架构及以后的 GPU，使用 `-G` 标志即可。

## 2 Nsight Systems 使用详解

### 2.1 工具概述

Nsight Systems 是一个系统级的性能分析工具，能够对整个应用程序进行时间线分析。它可以显示 CPU 和 GPU 上所有活动的交互视图，包括 CUDA 内核执行、内存传输、API 调用、同步事件等。通过时间线视图，开发者可以直观地了解程序的执行流程，找出 CPU-GPU 异步执行中的问题以及潜在的并行化机会。

### 2.2 基本使用方法

Nsight Systems 的命令行工具是 `nsys`。最基本的使用方法只需要指定要分析的可执行文件及其参数：

```bash
# 基本用法
nsys profile ./profile_cuda_gemm_fp32

# 指定输出文件名（不带扩展名）
nsys profile --output=gemm_profile ./profile_cuda_gemm_fp32

# 设置输出文件和报告类型
nsys profile --output=gemm_report --report=profile ./profile_cuda_gemm_fp32
```

运行上述命令后，nsys 会在当前目录生成一个 `.nsys-rep` 文件。这个文件可以使用 Nsight Systems GUI 打开进行交互式分析。默认情况下，输出文件名为 `report1.nsys-rep`，每次运行会自动递增编号。

### 2.3 常用命令行选项

`nsys` 命令行工具提供多种命令，其中最常用的是 `profile` 命令。以下是完整的选项说明：

#### 2.3.1 nsys 子命令

```bash
nsys --version                                  # 查看版本
nsys --help                                     # 查看帮助

# 核心命令：
nsys profile <program>                          # 运行程序并分析，生成 .nsys-rep 文件
nsys launch <program>                           # 启动程序准备分析
nsys start                                      # 开始分析会话
nsys stop                                       # 停止分析并保存结果
nsys cancel                                     # 取消分析并丢弃数据
nsys service                                    # 启动数据服务
nsys stats <file.nsys-rep>                      # 从报告生成统计信息
nsys status                                     # 查看当前分析环境状态
nsys shutdown                                   # 断开连接并关闭分析器
nsys sessions list                              # 列出活跃的分析会话
nsys export <input.nsys-rep> <output>           # 导出为其他格式
nsys analyze <file.nsys-rep>                    # 识别优化机会
nsys recipe                                     # 运行多节点分析脚本
nsys nvprof <args>                              # 转换 nvprof 参数并执行
```

#### 2.3.2 nsys profile 常用选项

```bash
# 1. 指定输出文件
nsys profile --output=my_profile ./program

# 2. 指定跟踪的 CUDA 事件类型
nsys profile --trace=cuda ./program             # 跟踪 CUDA API
nsys profile --trace=opengl,vulkan ./program    # 跟踪图形 API
nsys profile --trace=osrt ./program             # 跟踪 OS 运行时库
nsys profile --trace=cuda,osrt,nvtx ./program   # 组合多个选项

# 3. 设置采样频率（单位：Hz）
nsys profile --sample=cpu --sampling-frequency=1000 ./program

# 4. 排除特定库或文件的跟踪
nsys profile --exclude-cuda-rt ./program        # 排除 CUDA 运行时
nsys profile --exclude-path=/usr/lib ./program  # 排除特定路径

# 5. 设置跟踪缓冲区大小（单位：MB）
nsys profile --buffer-size=128 ./program

# 6. 在 GPU 上启用性能计数器收集
nsys profile --gpu-metrics-device=0 ./program

# 7. 设置最大跟踪文件大小（单位：MB）
nsys profile --max-mem=512 ./program

# 8. 设置要分析的内核数量限制
nsys profile --max-kernels=100 ./program

# 9. 延迟启动分析（等待指定秒数后开始）
nsys profile --delay=5 ./program

# 10. 限制分析持续时间（指定秒数后停止）
nsys profile --duration=30 ./program

# 11. 打印详细的启动信息
nsys profile --verbose=true ./program

# 12. 使用后台模式运行
nsys profile --background=true ./program

# 13. 跟踪 NVTX 标记的区域
nsys profile --trace=cuda,nvtx ./program

# 14. 排除特定进程
nsys profile --trace=cuda --exclude-path=/usr/lib ./program

# 15. 收集 CPU 调用栈
nsys profile --trace=cuda --call-stack ./program
```

#### 2.3.3 nsys stats 报告生成

```bash
# 生成概要报告
nsys stats --report summary ./gemm_profile.nsys-rep

# 生成 CUDA API 统计
nsys stats --report cuda_api ./gemm_profile.nsys-rep

# 生成 CUDA 内核统计
nsys stats --report cuda_kernel ./gemm_profile.nsys-rep

# 生成 GPU 指标统计
nsys stats --report gpu_metrics ./gemm_profile.nsys-rep

# 生成所有可用报告类型
nsys stats --report all ./gemm_profile.nsys-rep

# 指定输出目录
nsys stats --report cuda_kernel --output-dir=./analysis_results ./gemm_profile.nsys-rep

# 导出为 CSV 格式
nsys stats --report cuda_kernel --format csv --output-dir=./analysis_results ./gemm_profile.nsys-rep
```

### 2.4 针对 GEMM 项目的具体使用示例

以下是在 GEMM 项目中使用 Nsight Systems 的具体命令：

```bash
# 进入构建目录
cd /path/to/LearningCUDA/build

# 基本分析命令
nsys profile ./GEMM/profile_cuda_gemm_fp32

# 收集 GPU 性能计数器（推荐用于 GPU 性能分析）
nsys profile --gpu-metrics-device=0 --output=gemm_gpu_metrics ./GEMM/profile_cuda_gemm_fp32

# 跟踪 CUDA API 和 GPU 活动
nsys profile --trace=cuda --gpu-metrics-device=0 --output=gemm_cuda_trace ./GEMM/profile_cuda_gemm_fp32

# 完整功能分析
nsys profile --trace=cuda,osrt --gpu-metrics-device=0 --buffer-size=256 --output=gemm_full_profile ./GEMM/profile_cuda_gemm_fp32

# 多次运行取平均值（用于更稳定的性能数据）
for i in {1..5}; do
    nsys profile --gpu-metrics-device=0 --output=gemm_run_${i} ./GEMM/profile_cuda_gemm_fp32
done
```

### 2.5 生成分析报告

除了生成交互式的 `.nsys-rep` 文件外，nsys 还可以直接生成文本报告，这对于自动化分析和比较非常有用：

```bash
# 生成概要报告
nsys stats --report summary ./gemm_profile.nsys-rep

# 生成 CUDA API 统计
nsys stats --report cuda_api ./gemm_profile.nsys-rep

# 生成 CUDA 内核统计
nsys stats --report cuda_kernel ./gemm_profile.nsys-rep

# 生成 GPU 指标统计
nsys stats --report gpu_metrics ./gemm_profile.nsys-rep

# 生成所有可用的报告类型
nsys stats --report all ./gemm_profile.nsys-rep

# 指定输出目录
nsys stats --report cuda_kernel --output-dir=./analysis_results ./gemm_profile.nsys-rep

# 将报告保存为 CSV 格式
nsys stats --report cuda_kernel --format csv --output-dir=./analysis_results ./gemm_profile.nsys-rep
```

### 2.6 交互式分析

Nsight Systems 提供了图形界面，可以打开生成的 `.nsys-rep` 文件进行交互式分析：

```bash
# 在 Linux 上启动 GUI（需要 X11 转发或本地显示）
nsight-systems

# 或者直接通过文件关联打开
nsight-systems ./gemm_profile.nsys-rep
```

在 GUI 中，可以进行以下操作：

```text
时间线视图操作：
- 缩放：鼠标滚轮或使用 +/- 键
- 平移：拖动时间线或使用方向键
- 定位到特定事件：双击事件或右键选择 "Go To"

CPU 线程视图：
- 显示每个 CPU 线程的活动
- 识别空闲线程和同步等待
- 查看函数调用栈

CUDA 视图：
- 查看内核执行时间线
- 分析内存传输操作
- 识别并行执行机会

GPU 指标视图：
- 查看 GPU 利用率
- 分析内存带宽使用
- 识别计算和内存瓶颈
```

## 3 Nsight Compute 使用详解

### 3.1 工具概述

Nsight Compute 是一个专业的 CUDA 内核性能分析工具，提供深入的 GPU 硬件计数器信息。与 Nsight Systems 的系统级视图不同，Nsight Compute 专注于单个内核的分析，能够提供寄存器使用、共享内存使用、 occupancy（占用率）、内存访问模式等详细信息。它还包含一个规则引擎，可以自动识别性能问题并给出优化建议。

### 3.2 基本使用方法

Nsight Compute 的命令行工具是 `ncu`。以下是完整的命令行选项说明：

#### 3.2.1 启动模式

```bash
# 基本用法（分析所有内核）
ncu ./profile_cuda_gemm_fp32

# 指定输出文件
ncu --output=gemm_kernel_analysis.ncu-rep ./profile_cuda_gemm_fp32

# 只分析特定的内核（支持通配符）
ncu --kernel-name="gemm_kernel_v01*" --output=gemm_v01.ncu-rep ./profile_cuda_gemm_fp32

# 分析第 N 次内核启动（从 1 开始计数）
ncu --kernel-id=1 --output=kernel_1st.ncu-rep ./profile_cuda_gemm_fp32

# 交互模式选择
ncu --mode=launch-and-attach ./program     # 启动并附加分析（默认）
ncu --mode=launch ./program                # 启动程序，稍后附加
ncu --mode=attach --hostname 127.0.0.1     # 附加到已启动的程序
```

#### 3.2.2 通用选项

```bash
-h [ --help ]                              # 打印帮助信息
-v [ --version ]                           # 打印版本号
--mode arg (=launch-and-attach)            # 交互模式：
                                             #   launch-and-attach - 启动并附加分析（默认）
                                             #   launch - 启动并暂停，等待附加
                                             #   attach - 附加到已启动的程序
-p [ --port ] arg (=49152)                 # 连接目标应用程序的基准端口
--max-connections arg (=64)                # 连接目标应用程序的最大端口数
--config-file arg (=1)                     # 使用 config.ncu-cfg 配置文件
--config-file-path arg                     # 覆盖配置文件默认路径
--quiet                                    # 抑制所有输出
--verbose                                  # 更详细的输出
--log-file arg                             # 将输出写入指定文件
-f [ --force-overwrite ]                   # 强制覆盖所有输出文件
-i [ --import ] arg                        # 读取现有报告进行分析
--open-in-ui                               # 在 UI 中打开报告
```

#### 3.2.3 MPS（多进程服务）选项

```bash
# MPS 行为选择
--mps arg (=none)                          # MPS 模式：
                                             #   none - 无 MPS 支持（默认）
                                             #   client - 启动 MPS 客户端进程
                                             #   primary-client - 启动主 MPS 客户端进程
                                             #   control - 启动 MPS ncu 控制
--mps-num-clients arg                      # MPS 客户端进程数量
--mps-timeout arg                          # 发现 MPS 客户端进程的超时时间（秒）
```

#### 3.2.4 启动和附加选项

```bash
--check-exit-code arg (=1)                 # 检查程序退出码，非0则报错
--forward-signals                          # 将信号转发到应用程序根进程
--injection-path-32 arg                    # 覆盖 32 位注入库路径
--injection-path-64 arg                    # 覆盖 64 位注入库路径
--preload-library arg                      # 预加载共享库
--call-stack                               # 启用 CPU 调用栈收集
--call-stack-type arg                      # 调用栈类型：native（默认）, python
--nvtx                                     # 启用 NVTX 支持
--support-32bit                            # 支持分析 32 位应用程序
--target-processes arg (=all)              # 选择要分析的进程：
                                             #   application-only - 仅应用程序进程
                                             #   all - 应用程序及其子进程（默认）
--target-processes-filter arg              # 进程过滤器（逗号分隔）：
                                             #   <name> - 精确匹配进程名
                                             #   regex:<expr> - 正则表达式匹配
                                             #   exclude:<name> - 排除进程
                                             #   exclude-tree:<name> - 排除进程及其子进程
--null-stdin                               # 使用 /dev/null 作为标准输入
--nvtx-push-pop-scope arg (=thread)        # NVTX push/pop 范围：
                                             #   thread - 每个线程独立（默认）
                                             #   process - 整个进程共享
--hostname arg                             # 设置连接目标的主机名或 IP 地址
```

### 3.3 收集指标和事件

#### 3.3.1 指标收集选项

```bash
# 1. 收集所有指标（完整分析，最慢但最全面）
ncu --metrics all ./profile_cuda_gemm_fp32

# 2. 收集特定类别指标
# 收集计算吞吐量相关指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained ./profile_cuda_gemm_fp32

# 收集内存相关指标
ncu --metrics dram__bytes.sum ./profile_cuda_gemm_fp32

# 收集占用率相关指标
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained ./profile_cuda_gemm_fp32

# 3. 收集多个指标（逗号分隔）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained,dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained ./profile_cuda_gemm_fp32

# 4. 使用指标集合名称
ncu --metrics collection=SpeedOfLight ./profile_cuda_gemm_fp32
ncu --metrics collection=Memory ./profile_cuda_gemm_fp32
ncu --metrics collection=Launch ./profile_cuda_gemm_fp32

# 5. 使用正则表达式匹配指标
ncu --metrics regex:sm__throughput.* ./program

# 6. 使用指标组
ncu --metrics group:SpeedOfLight ./program
```

#### 3.3.2 事件计数器收集

```bash
# 收集事件计数器
ncu --events elapsed_cycles_sm,inst_executed ./profile_cuda_gemm_fp32

# 同时收集指标和事件
ncu --metrics sm__throughput.avg.pct_of_peak_sustained --events elapsed_cycles_sm ./profile_cuda_gemm_fp32
```

#### 3.3.3 指标查询

```bash
# 列出所有可收集的指标
ncu --list-metrics

# 查询系统可用指标
ncu --query-metrics

# 查询特定设备的指标
ncu --query-metrics --devices 0

# 使用后缀模式查询
ncu --query-metrics --query-metrics-mode=suffix

# 查询所有指标名称（包括完整路径）
ncu --query-metrics --query-metrics-mode=all

# 查询指标集合
ncu --query-metrics-collection groups       # 查询指标组
ncu --query-metrics-collection device       # 查询设备属性
ncu --query-metrics-collection launch       # 查询启动属性
ncu --query-metrics-collection numa         # 查询 NUMA 拓扑
ncu --query-metrics-collection nvlink       # 查询 NVLink 指标
ncu --query-metrics-collection pmsampling   # 查询 PM 采样指标
ncu --query-metrics-collection profiling    # 查询分析指标（默认）
ncu --query-metrics-collection source       # 查询源指标
ncu --query-metrics-collection stats        # 查询统计指标
ncu --query-metrics-collection warpsampling # 查询 Warp 采样指标

# 列出所有支持的芯片类型
ncu --list-chips

# 指定芯片类型查询指标
ncu --query-metrics --chips=sm_80,sm_86 ./program
```

### 3.4 关键指标详解

以下是 GEMM 优化中最常用的性能指标：

```bash
# ==================== 计算性能指标 ====================

# SM 利用率（目标：接近 100%）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained ./program

# 指令吞吐量
ncu --metrics sm__instruction_throughput.avg.pct_of_peak_sustained ./program

# 每个内核的指令数
ncu --metrics sm__inst_executed.sum ./program

# ==================== 内存性能指标 ====================

# 全局内存读取字节数
ncu --metrics gld__bytes.sum ./program

# 全局内存写入字节数
ncu --metrics gld__bytes.sum,gst__bytes.sum ./program

# 全局内存吞吐量（字节/秒）
ncu --metrics dram__bytes.sum ./program

# 共享内存使用量（字节）
ncu --metrics sm__shared_store_bytes.sum,sm__shared_load_bytes.sum ./program

# L1/L2 缓存命中率
ncu --metrics sm__l1_cache_global_hit_rate.pct ./program

# ==================== 占用率指标 ====================

# 理论占用率
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained ./program

# 活跃线程数
ncu --metrics sm__threads_active.max ./program

# 每个线程束的寄存器和共享内存使用
ncu --metrics sm__regfile_hybrid_peak_operand_per_warp.pct \
     sm__shared_peak_operand_per_warp.pct ./program

# ==================== 数学性能指标 ====================

# FP32 吞吐量
ncu --metrics sm__sass_fp32_ops.sum ./program

# Tensor Core 使用情况（仅限支持 Tensor Core 的 GPU）
ncu --metrics sm__tensor_core_hmma_cycles.sum ./program

# ==================== 延迟和效率指标 ====================

# 内核执行时间
ncu --metrics gpu__time_duration ./program

# 内存访问效率
ncu --metrics gld_efficiency, gst_efficiency ./program
```

### 3.5 内核过滤选项

```bash
# 按内核名称精确匹配
ncu --kernel-name="gemm_kernel_v00" ./program

# 按正则表达式匹配
ncu --kernel-name="regex:gemm.*v0[01].*" ./program

# 按内核 ID 匹配（格式：context-id:stream-id:[name-operator:]kernel-name:invocation-nr）
ncu --kernel-id="::gemm_kernel:1" ./program

# 设置内核名称基准（用于 --kernel-name 和 --kernel-id）
# --kernel-name-base arg (=function)
ncu --kernel-name-base=function ./program    # 函数名（默认）
ncu --kernel-name-base=demangled ./program   # demangled 名称
ncu --kernel-name-base=mangled ./program     # mangled 名称

# 重命名内核（使用配置文件）
ncu --rename-kernels ./program               # 启用内核重命名
ncu --rename-kernels-export ./program        # 导出简化名称到配置文件
ncu --rename-kernels-path=/path/to/config ./program  # 指定配置文件路径

# 限制收集的内核数量
ncu --launch-count=5 ./program

# 跳过前 N 次内核启动（仅匹配的内核）
ncu --launch-skip=3 ./program

# 跳过所有启动（包括不匹配的）
ncu --launch-skip-before-match=3 ./program

# 过滤模式
ncu --filter-mode=global ./program          # 全局应用（默认）
ncu --filter-mode=per-gpu ./program         # 按 GPU 分别应用
ncu --filter-mode=per-launch-config ./program # 按启动配置分别应用

# 指定分析的 GPU 设备
ncu --devices=0,1 ./program

# NVTX 范围过滤
ncu --nvtx --nvtx-include="Domain A@Range A" ./program
ncu --nvtx --nvtx-exclude="Range A" ./program

# NVTX 范围过滤器（更复杂的过滤）
# --range-filter=<yes/no/on/off>:<start/stop range instances>:<NVTX range instances>
ncu --range-filter=yes:1:1-3 ./program       # 过滤指定实例

# CPU 调用栈过滤
ncu --native-include="Module A@File A@Function A" ./program
ncu --native-exclude="File A@Function A" ./program

# Python 调用栈过滤
ncu --python-include="Module A@File A@Function A" ./program
ncu --python-exclude="File A@Function A" ./program

# 收集特定截面
ncu --section=section_identifier ./program
ncu --section=regex:Memory.* ./program       # 正则表达式匹配

# 禁用额外后缀收集（avg, min, max, sum）
ncu --disable-extra-suffixes ./program
```

### 3.6 重放模式选项

```bash
# 内核重放模式（默认）- 单独重放每个内核
ncu --replay-mode=kernel ./program

# 应用程序重放模式 - 重新运行整个应用程序
ncu --replay-mode=application ./program

# 范围重放模式 - 重放指定范围的启动
ncu --replay-mode=range ./program

# 应用程序范围重放模式
ncu --replay-mode=app-range ./program

# 应用程序重放缓冲区位置
ncu --app-replay-buffer=file ./program    # 使用临时文件（默认）
ncu --app-replay-buffer=memory ./program  # 使用内存

# 应用程序重放内核匹配策略
ncu --app-replay-match=name ./program     # 按名称匹配
ncu --app-replay-match=grid ./program     # 按名称和网格/块大小匹配（默认）
ncu --app-replay-match=all ./program      # 精确匹配所有属性

# 应用程序重放匹配模式
ncu --app-replay-mode=strict ./program    # 严格模式
ncu --app-replay-mode=balanced ./program  # 平衡模式（默认）
ncu --app-replay-mode=relaxed ./program   # 宽松模式

# 范围重放选项
ncu --range-replay-options=enable-greedy-sync ./program
ncu --range-replay-options=disable-host-restore ./program
ncu --range-replay-options=disable-host-save ./program
ncu --range-replay-options=disable-dependent-kernel-detection ./program

# CUDA 图分析模式
ncu --graph-profiling=node ./program       # 分析单个内核节点（默认）
ncu --graph-profiling=graph ./program      # 分析整个图

# 终止目标应用程序（收集指定数量后）
ncu --kill=1 ./program

# 从程序开始分析
ncu --profile-from-start=yes ./program     # 是（默认）
ncu --profile-from-start=no ./program      # 否

# 禁用启动/停止 API
ncu --disable-profiler-start-stop ./program

# 导入 ELF cubins（包含 SASS, PTX 和元信息）
ncu --import-sass=yes ./program            # 导入（默认）
ncu --import-sass=no ./program             # 不导入

# 导入 CUDA 源文件
ncu --import-source=yes ./program          # 导入源文件
ncu --import-source=no ./program           # 不导入（默认）

# 源文件搜索路径
ncu --source-folders=/path/to/source1,/path/to/source2 ./program
```

### 3.7 报告和输出选项

```bash
# 1. 生成文本报告
ncu --report all --output-file=gemm_report.txt ./profile_cuda_gemm_fp32

# 2. 生成 CSV 格式报告
ncu --metrics sm__throughput.avg.pct_of_peak_sustained \
    --csv --print-units --output-file=gemm_metrics.csv \
    ./profile_cuda_gemm_fp32

# 3. 打印详细信息到控制台
ncu --metrics all --print-details --kernel-name="gemm_kernel_v00*" ./profile_cuda_gemm_fp32

# 4. 导出为 JSON 格式
ncu --metrics collection=SpeedOfLight \
    --export=gemm_analysis.json \
    ./profile_cuda_gemm_fp32

# 5. 使用 Nsight Compute GUI 进行交互式分析
ncu-ui ./gemm_analysis.ncu-rep

# 6. 选择报告页面
ncu --page=details ./program              # 详细信息（默认）
ncu --page=raw ./program                  # 原始数据
ncu --page=source ./program               # 源代码
ncu --page=session ./program              # 会话信息

# 7. 打印源码视图
ncu --print-source=sass ./program         # SASS 视图
ncu --print-source=ptx ./program          # PTX 视图
ncu --print-source=cuda ./program         # CUDA 源码视图
ncu --print-source=cuda,sass ./program    # CUDA 和 SASS 视图

# 解析源文件路径
ncu --resolve-source-file=/path/to/source ./program  # 指定源文件搜索路径

# 8. 打印细节级别
ncu --print-details=header ./program      # 仅头部（默认）
ncu --print-details=body ./program        # 仅主体
ncu --print-details=all ./program         # 全部

# 9. 打印指标名称格式
ncu --print-metric-name=label ./program   # 标签（默认）
ncu --print-metric-name=name ./program    # 名称
ncu --print-metric-name=label-name ./program # 标签和名称

# 10. 打印单位格式
ncu --print-units=auto ./program          # 自动缩放（默认）
ncu --print-units=base ./program          # 基础单位

# 11. 打印浮点数格式
ncu --print-fp ./program

# 12. 打印内核名称基准
ncu --print-kernel-base=demangled ./program # demangled 名称（默认）
ncu --print-kernel-base=function ./program  # 函数名
ncu --print-kernel-base=mangled ./program   # mangled 名称

# 13. 打印指标归因
ncu --print-metric-attribution ./program   # 显示指标归因级别

# 14. 打印规则详情
ncu --print-rule-details ./program        # 打印规则详细信息

# 15. 打印摘要模式
ncu --print-summary=none ./program        # 无（默认）
ncu --print-summary=per-gpu ./program     # 按GPU
ncu --print-summary=per-kernel ./program  # 按内核
ncu --print-summary=per-nvtx ./program    # 按NVTX

# 16. 打印指标实例值
ncu --print-metric-instances=none ./program     # 仅汇总值（默认）
ncu --print-metric-instances=values ./program   # 包含所有实例值
ncu --print-metric-instances=details ./program  # 包含关联ID和实例值

# 17. NVTX 重命名
ncu --print-nvtx-rename=none ./program     # 不使用NVTX重命名（默认）
ncu --print-nvtx-rename=kernel ./program   # 使用NVTX重命名内核
```

### 3.8 使用规则引擎

Nsight Compute 内置了规则引擎，可以自动检测常见的性能问题：

```bash
# 启用所有规则检查
ncu --rules --kernel-name="gemm_kernel_v00*" ./profile_cuda_gemm_fp32

# 查看所有可用的规则
ncu --list-rules

# 运行特定规则
ncu --rules=MemoryRule,ComputeRule --kernel-name="gemm_kernel_v00*" ./profile_cuda_gemm_fp32

# 查看规则的详细信息
ncu --rule-details=MemoryRule

# 禁用规则应用
ncu --apply-rules=no ./program

# 选择特定规则
ncu --rule=MemoryRule ./program
```

常见的自动检测规则包括：

```text
1. Memory Rules
   - Global Memory Access Efficiency
   - Shared Memory Bank Conflicts
   - L1/L2 Cache Utilization

2. Compute Rules
   - SM Utilization
   - Instruction Efficiency
   - Warp Execution Efficiency

3. Occupancy Rules
   - Theoretical Occupancy
   - Active Warps per SM

4. Algorithm Rules
   - Memory Coalescing
   - Divergent Branching
```

### 3.9 硬件控制选项

```bash
# 缓存控制
ncu --cache-control=all ./program          # 启用所有缓存
ncu --cache-control=none ./program         # 禁用缓存

# GPU 时钟控制
ncu --clock-control=base ./program         # 锁定到基础时钟
ncu --clock-control=none ./program         # 不锁定时钟
ncu --clock-control=reset ./program        # 重置时钟并退出

# Tensor Core 加速状态
ncu --pipeline-boost-state=stable ./program  # 稳定模式（推荐）
ncu --pipeline-boost-state=dynamic ./program # 动态模式
```

### 3.10 针对 GEMM 项目的完整分析示例

以下是对 GEMM 内核进行完整性能分析的具体命令：

```bash
# 进入构建目录
cd /path/to/LearningCUDA/build

# 1. 快速概览（收集基础指标）
ncu --metrics collection=SpeedOfLight \
    --kernel-name="gemm_kernel_v00*" \
    --output=gemm_v00_overview ./GEMM/profile_cuda_gemm_fp32

# 2. 内存访问分析
ncu --metrics \
    gld__bytes.sum, \
    gst__bytes.sum, \
    gld__bytes.sum.per_second, \
    dram__bytes.sum, \
    dram__bytes.sum.per_second, \
    sm__l1_cache_global_hit_rate.pct, \
    sm__l2_cache_global_hit_rate.pct \
    --kernel-name="gemm_kernel_v00*" \
    --output=gemm_v00_memory ./GEMM/profile_cuda_gemm_fp32

# 3. 占用率分析
ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained, \
    sm__block_active.avg.pct_of_peak_sustained, \
    sm__threads_active.avg, \
    sm__occupancy.block_per_sm.max, \
    sm__occupancy.threads.max.per_sm \
    --kernel-name="gemm_kernel_v00*" \
    --output=gemm_v00_occupancy ./GEMM/profile_cuda_gemm_fp32

# 4. 完整分析（包含所有指标）
ncu --metrics all \
    --kernel-name="gemm_kernel_v00*" \
    --output=gemm_v00_full ./GEMM/profile_cuda_gemm_fp32

# 5. 比较两个内核版本
ncu --metrics collection=SpeedOfLight \
    --kernel-name="gemm_kernel_v00*" \
    --output=gemm_v00_compare ./GEMM/profile_cuda_gemm_fp32

ncu --metrics collection=SpeedOfLight \
    --kernel-name="gemm_kernel_v01*" \
    --output=gemm_v01_compare ./GEMM/profile_cuda_gemm_fp32

# 6. 自动规则检查
ncu --metrics all --rules \
    --kernel-name="gemm_kernel_v00*" \
    --output=gemm_v00_rules ./GEMM/profile_cuda_gemm_fp32

# 7. 使用 NVTX 范围过滤
ncu --nvtx --nvtx-include="GEMM" \
    --kernel-name="gemm_kernel_v00*" \
    --output=gemm_v00_nvtx ./GEMM/profile_cuda_gemm_fp32
```

### 3.11 采样和性能监控选项

```bash
# PM 采样间隔
ncu --pm-sampling-interval=0 ./program        # 自动确定
ncu --pm-sampling-interval=1000 ./program     # 指定周期数

# PM 采样缓冲区大小
ncu --pm-sampling-buffer-size=0 ./program     # 自动确定

# PM 最大采样次数
ncu --pm-sampling-max-passes=0 ./program      # 自动确定

# Warp 采样间隔（0-31，实际频率 2^(5+value) 周期）
ncu --warp-sampling-interval=auto ./program   # 自动确定
ncu --warp-sampling-interval=10 ./program     # 指定值

# Warp 最大采样次数
ncu --warp-sampling-max-passes=5 ./program    # 默认值

# Warp 采样缓冲区大小
ncu --warp-sampling-buffer-size=33554432 ./program
```

### 3.12 截面和集合选项

```bash
# 列出所有截面集合
ncu --list-sets

# 收集特定截面集合
ncu --set=section_set_name ./program

# 列出所有可用截面
ncu --list-sections

# 收集特定截面
ncu --section=section_identifier ./program
ncu --section=regex:Memory.* ./program       # 正则表达式匹配

# 截面搜索路径
ncu --section-folder=/path/to/sections ./program
ncu --section-folder-recursive=/path/to ./program

# 恢复默认截面文件夹
ncu --section-folder-restore
```

## 4 高级使用技巧

### 4.1 自动化批量分析

对于需要多次运行或分析多个配置的场景，可以使用脚本进行自动化：

```bash
#!/bin/bash
# analyze_gemm.sh - GEMM 性能分析脚本

EXECUTABLE="./GEMM/profile_cuda_gemm_fp32"
OUTPUT_DIR="./profiling_results"

mkdir -p $OUTPUT_DIR

# 收集不同矩阵大小的数据
for size in 512 1024 2048; do
    echo "分析矩阵大小: $size"

    # 使用环境变量设置矩阵大小（如果程序支持）
    MATRIX_SIZE=$size ncu \
        --metrics collection=SpeedOfLight \
        --output=$OUTPUT_DIR/gemm_${size}_ncu.ncu-rep \
        $EXECUTABLE

    nsys profile \
        --gpu-metrics-device=0 \
        --output=$OUTPUT_DIR/gemm_${size}_nsys \
        $EXECUTABLE
done

# 批量生成报告
for file in $OUTPUT_DIR/*.ncu-rep; do
    filename=$(basename $file .ncu-rep)
    ncu --report all --output-file=${OUTPUT_DIR}/${filename}_report.txt $file
done
```

### 4.2 使用正则表达式匹配内核名称

```bash
# 分析所有以 "gemm" 开头的内核
ncu --kernel-name="gemm.*" ./program

# 分析包含特定模式的内核
ncu --kernel-name=".*v0[01].*" ./program

# 精确匹配
ncu --kernel-name="^gemm_kernel_v00$" ./program
```

### 4.3 多 GPU 配置

```bash
# 指定分析的 GPU 设备
ncu --device=0 --kernel-name="gemm.*" ./program

# 分析多个 GPU（需要程序支持）
ncu --devices=0,1 --output=gpu_analysis ./program

# 在 Docker 容器中使用（需要正确配置 GPU 访问）
docker run --gpus all --rm -v $(pwd):/workspace \
    nvcr.io/nvidia/cuda:12.0-base-ubuntu22.04 \
    ncu --output=/workspace/analysis.ncu-rep ./program
```

### 4.4 比较不同优化版本

```bash
# 收集 Naive 版本数据
ncu --metrics collection=SpeedOfLight \
    --kernel-name="gemm_kernel_v00*" \
    --export=naive_v00.json \
    ./profile_cuda_gemm_fp32

# 收集 Shared Memory 版本数据
ncu --metrics collection=SpeedOfLight \
    --kernel-name="gemm_kernel_v01*" \
    --export=shared_v01.json \
    ./profile_cuda_gemm_fp32

# 使用 Python 脚本比较结果
python3 << 'EOF'
import json

def load_and_compare(file1, file2):
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    # 比较 SM 利用率
    sm1 = data1['metrics'][0]['value']
    sm2 = data2['metrics'][0]['value']
    print(f"SM 利用率: {sm1:.1f}% -> {sm2:.1f}% (提升: {(sm2-sm1)/sm1*100:.1f}%)")

load_and_compare('naive_v00.json', 'shared_v01.json')
EOF
```

## 5 结果解读指南

### 5.1 Nsight Systems 结果解读

```text
时间线分析关键点：

1. CPU-GPU 重叠度
   - 理想情况：CPU 和 GPU 的执行条重叠
   - 问题：GPU 空闲等待 CPU 提交工作
   - 解决方案：使用异步内存传输和流

2. 内存传输时间占比
   - 理想情况：计算时间远大于传输时间
   - 问题：大量时间用于数据传输
   - 解决方案：增加数据重用，使用页锁定内存

3. 内核执行模式
   - 串行：内核一个接一个执行
   - 并行：多个内核同时执行
   - 优化：使用 CUDA 流实现并行执行

4. 线程利用率
   - 检查 CPU 线程是否都忙碌
   - 识别同步点和等待时间
```

### 5.2 Nsight Compute 结果解读

```text
关键性能指标解读：

1. SM Utilization (SM 利用率)
   - 目标：> 90%
   - 低值可能原因：
     * 内存带宽限制
     * 同步点过多
     * 内核太简单

2. Memory Throughput (内存吞吐量)
   - 与理论峰值比较
   - 公式：实际吞吐量 / 理论峰值 * 100%
   - 低值可能原因：
     * 内存访问不合并
     * 缓存未命中率高

3. Occupancy (占用率)
   - 目标：> 50%（具体取决于内核特性）
   - 低值可能原因：
     * 寄存器使用过多
     * 共享内存使用过多
     * 每个线程块线程数过少

4. L1/L2 Cache Hit Rate
   - 目标：> 80%
   - 低值可能原因：
     * 访问模式分散
     * 数据重用率低
```

## 6 常见问题与解决方案

### 6.1 工具安装问题

```bash
# 问题：ncu 或 nsys 命令未找到
# 解决：确认 CUDA Toolkit 安装路径
export PATH=/usr/local/cuda/bin:$PATH
# 或检查安装位置
which ncu
which nsys

# 问题：权限被拒绝
# 解决：添加执行权限
chmod +x /usr/local/cuda/bin/nsys
chmod +x /usr/local/cuda/bin/ncu

# 问题：GPU 计数器不可用
# 解决：使用 root 权限运行
sudo ncu --metrics all ./program

# 问题：CUDA 版本不兼容
# 解决：检查 CUDA 版本并安装兼容版本
nvcc --version
nvidia-smi

# 问题：Nsight Compute 和 CUDA Toolkit 版本不匹配
# 解决：确保使用匹配的版本
ncu --version
```

### 6.2 分析过程中的问题

```bash
# 问题：分析输出为空
# 解决：检查可执行文件是否正确
ls -la ./profile_cuda_gemm_fp32
file ./profile_cuda_gemm_fp32

# 问题：内核未被识别
# 解决：使用通配符或检查内核名称
ncu --kernel-name=".*" ./program

# 问题：GPU 指标收集失败
# 解决：指定 GPU 设备并检查驱动
nvidia-smi
ncu --devices=0 ./program

# 问题：调试信息缺失
# 解决：使用调试模式重新编译
cmake .. -DCMAKE_CUDA_FLAGS="-g -G"

# 问题：指标收集失败
# 解决：查询可用指标
ncu --query-metrics
ncu --list-metrics

# 问题：规则检查失败
# 解决：列出可用规则
ncu --list-rules

# 问题：截面收集失败
# 解决：列出可用截面
ncu --list-sections
ncu --list-sets

# 问题：内核重放失败
# 解决：使用应用重放模式
ncu --replay-mode=application ./program

# 问题：NVTX 范围过滤不工作
# 解决：启用 NVTX 支持
ncu --nvtx --nvtx-include="Range Name" ./program

# 问题：内存不足
# 解决：减少重放次数或使用文件缓冲区
ncu --app-replay-buffer=file ./program
ncu --launch-count=1 ./program
```

### 6.3 nsys 常见问题

```bash
# 问题：nsys 报告生成失败
# 解决：使用 stats 命令单独生成报告
nsys stats report1.nsys-rep

# 问题：GPU 指标不可用
# 解决：启用 GPU 指标收集
nsys profile --gpu-metrics-device=0 ./program

# 问题：跟踪缓冲区溢出
# 解决：增加缓冲区大小
nsys profile --buffer-size=256 ./program

# 问题：跟踪文件过大
# 解决：限制跟踪内容或持续时间
nsys profile --max-mem=512 --duration=30 ./program

# 问题：无法附加到正在运行的进程
# 解决：使用 launch 模式
nsys launch ./program
nsys start
nsys stop
```

### 6.4 ncu 常见问题

```bash
# 问题：无法连接到目标进程
# 解决：检查端口和主机名
ncu --mode=attach --hostname 127.0.0.1 --port 49152

# 问题：报告导入失败
# 解决：检查文件路径和格式
ncu --import report.ncu-rep

# 问题：调用栈收集失败
# 解决：启用调用栈收集
ncu --call-stack ./program

# 问题：配置文件加载失败
# 解决：指定配置文件路径
ncu --config-file-path=/path/to/config.ncu-cfg ./program

# 问题：内核重命名不工作
# 解决：检查配置文件
ncu --rename-kernels-export=1 ./program

# 问题：指标需要后缀
# 解决：使用完整指标名称或 --query-metrics
ncu --metrics regex:sm__.* ./program
```
