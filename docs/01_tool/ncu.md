# NVIDIA Nsight Compute

[ncu下载](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)  [User Guide](https://docs.nvidia.com/nsight-compute/)

#### 主要功能

- **内核剖析**：提供每个CUDA内核的执行时间、内存带宽、指令吞吐量等详细指标。
- **优化建议**：根据内核的执行效率，提供具体的优化建议，如内存访问优化、指令调度优化等。
- **调试支持**：支持内核调试，帮助开发者定位和修复代码中的错误。

[相关指令信息](../data/ncu_help)

## 常用命令

```shell
# 基础分析
ncu ./your_application

# 分析指定内核
ncu -k kernel_name ./your_application

# 分析前2次内核启动
ncu -k foo -c 2 ./your_application

# 导出报告
ncu -o report_name ./your_application

# 加载现有报告
ncu --import myReport.ncu-rep
```

## General Options

| 参数 | 说明 |
|------|------|
| `-h, --help` | 打印帮助信息 |
| `-v, --version` | 打印版本号 |
| `--mode` | 交互模式：launch-and-attach, launch, attach |
| `-p, --port` | 连接目标应用的端口，默认49152 |
| `--max-connections` | 最大连接数，默认64 |
| `--config-file` | 配置文件名，默认1（搜索当前目录和$HOME/.config/NVIDIA Corporation） |
| `--config-file-path` | 覆盖配置文件的默认路径 |

## MPS Options

| 参数 | 说明 |
|------|------|
| `--mps` | MPS行为：none, client, primary-client, control |
| `--mps-num-clients` | MPS客户端进程数 |
| `--mps-timeout` | 发现MPS客户端进程的超时时间（秒） |

## Launch Options

| 参数 | 说明 |
|------|------|
| `--check-exit-code` | 检查应用退出码，默认1（开启） |
| `--forward-signals` | 转发所有信号给应用根进程 |
| `--injection-path-32` | 32位注入库路径 |
| `--injection-path-64` | 64位注入库路径 |
| `--preload-library` | 预加载共享库 |
| `--call-stack` | 启用CPU调用栈收集 |
| `--call-stack-type` | 调用栈类型：native(默认), python |
| `--nvtx` | 启用NVTX支持 |
| `--support-32bit` | 支持从32位应用启动的分析 |
| `--target-processes` | 目标进程：application-only, all |
| `--target-processes-filter` | 进程过滤器：`<process name>`, `regex:<expression>`, `exclude:<name>`, `exclude-tree:<name>` |
| `--null-stdin` | 使用/dev/null作为标准输入 |
| `--nvtx-push-pop-scope` | NVTX push/pop范围：thread(默认), process |

## Attach Options

| 参数 | 说明 |
|------|------|
| `--hostname` | 连接目标的hostname或IP地址 |

## Common Profile Options

| 参数 | 说明 |
|------|------|
| `--kill` | 分析指定数量后终止应用，默认0（不终止） |
| `--replay-mode` | 重放机制：kernel(默认), application, range, app-range |
| `--app-replay-buffer` | 应用重放缓冲区位置：file(默认), memory |
| `--app-replay-match` | 内核匹配策略：name, grid(默认), all |
| `--app-replay-mode` | 匹配模式：strict, balanced(默认), relaxed |
| `--graph-profiling` | CUDA图分析模式：node(默认), graph |
| `--range-replay-options` | 范围重放选项：enable-greedy-sync, disable-host-restore, disable-host-save, disable-dependent-kernel-detection |
| `--list-sets` | 列出所有section sets |
| `--set` | 要收集的section set标识符 |
| `--list-sections` | 列出所有sections |
| `--section-folder` | section文件的搜索路径（非递归） |
| `--section-folder-recursive` | section文件的搜索路径（递归） |
| `--section-folder-restore` | 恢复stock文件到默认section文件夹 |
| `--list-rules` | 列出所有分析规则 |
| `--apply-rules` | 应用分析规则：on/off, yes/no |
| `--rule` | 要应用的分析规则标识符 |
| `--import-sass` | 导入ELF cubins到报告：on(默认)/off, yes/no |
| `--import-source` | 导入CUDA源文件：on/off, yes(默认)/no |
| `--source-folders` | CUDA源文件的搜索路径（递归） |
| `--list-metrics` | 列出所有要收集的指标 |
| `--query-metrics` | 查询可用指标 |
| `--query-metrics-mode` | 查询指标模式：base(默认), suffix, all |
| `--query-metrics-collection` | 查询指标类型：device, groups, launch, numa, nvlink, pmsampling, profiling(默认), source, stats, warpsampling |
| `--list-chips` | 列出所有支持的芯片 |
| `--chips` | 指定查询指标的芯片 |
| `--profile-from-start` | 从应用开始分析：on(默认)/off, yes/no |
| `--disable-profiler-start-stop` | 禁用cu(da)ProfilerStart/Stop API |
| `--quiet` | 抑制所有分析器输出 |
| `--verbose` | 增加分析器输出详细度 |
| `--cache-control` | GPU缓存行为：all, none |
| `--clock-control` | GPU时钟控制：base(锁定到基础频率), none(不锁定), reset(重置时钟并退出) |
| `--pipeline-boost-state` | Tensor Core boost状态：stable(默认), dynamic |

## Filter Profile Options

| 参数 | 说明 |
|------|------|
| `--devices` | 启用的设备ID列表（逗号分隔） |
| `--filter-mode` | 过滤模式：global(默认), per-gpu, per-launch-config |
| `--kernel-id` | 内核标识符，格式：`context-id:stream-id:[name-operator:]kernel-name:invocation-nr` |
| `-k, --kernel-name` | 内核名称过滤：`<kernel name>` 或 `regex:<expression>` |
| `--kernel-name-base` | 内核名称基准：function(默认), demangled, mangled |
| `--rename-kernels` | 重命名内核：on(默认)/off, yes/no |
| `--rename-kernels-export` | 导出重命名配置到文件 |
| `--rename-kernels-path` | 重命名配置文件的路径 |
| `-c, --launch-count` | 限制收集的分析结果数量 |
| `-s, --launch-skip` | 开始分析前跳过的内核启动次数，默认0 |
| `--launch-skip-before-match` | 开始分析前跳过的所有内核启动次数，默认0 |
| `--section` | 收集的section标识符，支持regex匹配 |
| `--metrics` | 指定要收集的指标（逗号分隔），支持regex:, group:, breakdown:前缀 |
| `--disable-extra-suffixes` | 禁用额外后缀收集（avg, min, max, sum） |
| `--nvtx-include` | NVTX包含过滤器 |
| `--nvtx-exclude` | NVTX排除过滤器 |
| `--range-filter` | NVTX范围过滤器，格式：`<yes/no/on/off>:<start/stop范围实例>:<NVTX范围实例>` |
| `--native-include` | 本地CPU调用栈包含过滤器 |
| `--native-exclude` | 本地CPU调用栈排除过滤器 |
| `--python-include` | Python调用栈包含过滤器 |
| `--python-exclude` | Python调用栈排除过滤器 |

## PM Sampling Options

| 参数 | 说明 |
|------|------|
| `--pm-sampling-interval` | PM采样间隔（周期或ns），0表示自动确定 |
| `--pm-sampling-buffer-size` | PM采样缓冲区大小（字节），0表示自动确定 |
| `--pm-sampling-max-passes` | PM采样最大pass数，0表示自动确定 |

## Warp State Sampling Options

| 参数 | 说明 |
|------|------|
| `--warp-sampling-interval` | Warp状态采样周期[0-31]，实际频率2^(5+value)，默认auto |
| `--warp-sampling-max-passes` | Warp状态采样最大pass数，默认5 |
| `--warp-sampling-buffer-size` | Warp状态采样缓冲区大小（字节），默认33554432 |

## File Options

| 参数 | 说明 |
|------|------|
| `--log-file` | 输出日志文件路径，stdout/stderr表示标准输出/错误 |
| `-o, --export` | 输出文件路径，不设置则使用临时文件 |
| `-f, --force-overwrite` | 强制覆盖所有输出文件 |
| `-i, --import` | 输入文件路径，用于读取分析结果 |
| `--open-in-ui` | 在UI中打开报告而非终端显示 |

## Console Output Options

| 参数 | 说明 |
|------|------|
| `--csv` | 使用逗号分隔值输出 |
| `--page` | 报告页面：details(默认), raw, source, session |
| `--print-source` | 源视图：sass, ptx, cuda, cuda,sass |
| `--resolve-source-file` | 源文件路径列表（逗号分隔） |
| `--print-details` | details页面输出内容：header(默认), body, all |
| `--print-metric-name` | 指标名称列显示：label(默认), name, label-name |
| `--print-units` | 指标单位缩放：auto(默认), base |
| `--print-metric-attribution` | 显示Green Context结果的归因级别 |
| `--print-fp` | 所有数值指标显示为浮点数 |
| `--print-kernel-base` | 内核名称输出基准：demangled(默认) |
| `--print-metric-instances` | 指标实例值输出模式：none(默认), values, details |
| `--print-nvtx-rename` | NVTX重命名方式：none(默认), kernel |
| `--print-rule-details` | 打印规则结果的附加详情 |
| `--print-summary` | 摘要输出模式：none, per-gpu, per-kernel, per-nvtx |

## 常用分析示例

```shell
# 基础分析指定内核
ncu -k matrixMul ./myApp

# 分析前5次内核启动
ncu -k foo -c 5 ./myApp

# 跳过前3次启动
ncu -k foo -s 3 ./myApp

# 导出CSV格式报告
ncu --csv -o report ./myApp

# 使用NVTX范围过滤
ncu --nvtx --nvtx-include "Domain A@Range A" ./myApp

# 加载现有报告
ncu --import myReport.ncu-rep

# 在UI中打开报告
ncu --import myReport.ncu-rep --open-in-ui

# 指定设备分析
ncu --devices 0,1 ./myApp

# 使用正则过滤内核名
ncu -k regex:^foo.* ./myApp

# 收集指定指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained ./myApp

# 收集指标组
ncu --metrics group:occupancy ./myApp
```

## NVTX过滤语法

```shell
# 分析Domain A中Range A内的内核
ncu --nvtx --nvtx-include "Domain A@Range A" ./myApp

# 排除Range A内的内核
ncu --nvtx --nvtx-exclude "Range A" ./myApp

# 组合过滤
ncu --nvtx --nvtx-include "Range A" --nvtx-exclude "Range B" ./myApp
```

## CPU调用栈过滤语法

```shell
# 本地调用栈过滤
ncu --native-include "Module A@File A@Function A" ./myApp

# Python调用栈过滤
ncu --python-include "Module A@File A@Function A" ./myApp

# 排除过滤器
ncu --native-exclude "Module B@File B@Function B" ./myApp
```

模块名和文件名是可选的，如果不提供将匹配所有模块和文件。



```shell
ncu --set full --kernel-name SlowKernel -o profile ./app

```



