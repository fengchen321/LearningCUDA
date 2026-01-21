# nvprof

```shell
nvprof --metrics branch_effiency ./demo  # 查看分支效率
# 可能为100%：编译器优化，将短的、有条件的代码段的断定指令取代了分支指令（导致分化的实际控制流指令）

nvprof --events branch,divergent_branch ./demo # 获得分支和分化分支的事件计数器

nvprof --metrics achieved_occupancy ./demo # 占用率
nvprof --metrics gld_throughtput ./demo # 检查内核的内存读取效率
nvprof --metrics gld_efficiency ./demo # 检查全局加载效率，被请求的全局加载吞吐量占所需全局加载吞吐量的比值，衡量加载操作利用设备内存带宽的程度
```

# Nvidia-smi 

```shell
nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" | grep "Memory" # 检测设备的内存频率

nvidia-smi --query-gpu=name,compute_cap --format=csv # 查看设备计算能力
```
