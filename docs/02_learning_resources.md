# CUDA 学习资源汇总

> 本项目相关的 CUDA 学习资料、教程和工具文档索引。

## 核心教程与项目

- [CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)：CUDA矩阵乘法优化实现代码库,测试代码框架参考
- [CUDA-GEMM-Optimization- 对应博客](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)：详细讲解GEMM优化技术的原理和实现

## 系统学习课程

- [CUDA-MODE：GPU编程与优化系列讲座资料](https://github.com/gpu-mode/lectures)
- [B站CUDA教程合集：中文视频教程，适合入门](https://space.bilibili.com/218427631/lists/4695308?type=series)
- [NVIDIA GPU C++教程：官方C++/CUDA学习资源](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/gpu-cpp-tutorial)
- [cuda-course：深度学习生态中的CUDA应用](https://github.com/Infatoshi/cuda-course/tree/master/01_Deep_Learning_Ecosystem)
- [cuda document](https://github.com/Infatoshi/docs.md/tree/master)

## 技术深度解析

- [Matmul优化博客：矩阵乘法优化技术详解](https://www.aleksagordic.com/blog/matmul)
- [知乎：CUDA优化经验分享：中文社区实践总结](https://zhuanlan.zhihu.com/p/707107808)
- [AMD Matrix Core优化：AMD GPU矩阵核心优化指南](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [gpu-glossary: GPU术语解释](https://developer.nvidia.com/gpu-glossarydal.com/gpu-glossary)

## 调试与性能分析

- [NVIDIA Nsight Compute文档：性能分析和调优工具指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA开发者论坛：共享内存bank冲突：解决共享内存bank冲突的讨论](https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900/12)
- [Performance_and_debugging_tools](https://github.com/FZJ-JSC/tutorial-multi-gpu/blob/main/04-L_Performance_and_debugging_tools/slides.pdf)
- [Nsight tool slide](https://live.nvidia.cn/gtc-od/attachments/CNS20632.pdf)

## 在线练习

- [LeetGPU: 在线练习](https://leetgpu.com/)

## 本地文档

本项目的相关文档：

- [安装环境配置](./00_Setting/install_environment.md)
- [GEMM 优化指南](./01_tool/cuda_gemm_optimization_guide.md)
- [NVIDIA 性能分析工具指南](./01_tool/nvidia_profiler_guide.md)
- [性能分析工具对比](./01_tool/nvprof_ncu_nsys.md)
- [GEMM 性能分析](./gemm_performance_analysis.md)
