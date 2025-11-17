# WSL2下 安装 CUDA 和 CUDNN

## 安装WSL2
```shell
wsl --install -d Ubuntu-22.04 # 安装后重启
wsl --shutdown # 使其stop
wsl --export Ubuntu-22.04 D:\wsl_ubuntu\Ubuntu.tar # 导出备份
wsl --unregister Ubuntu-22.04 #删除当前安装的系统
wsl --import Ubuntu-22.04 D:\wsl_ubuntu D:\wsl_ubuntu\Ubuntu.tar 
Ubuntu2204 config --default-user fengchen
```
## 配置C++环境（不同版本,学习cpp 可选）
```shell
# 安装 gcc
sudo apt install build-essential
# 添加 ppa 源
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
# 安装gcc-13和g++-13
sudo apt update
sudo apt install gcc-13 g++-13
# 设置 gcc-13 和 g++-13 优先级 数字越大优先级越高
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13
# 查看版本
gcc --version
g++ --version
# 查看优先级配置列表并切换
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
# 选择想用的版本即可
```
## 安装cuda
查看驱动版本： windows -> nivdia控制面板-> 帮助->系统信息->显示：驱动程序版本 560.94，组件：NVCUDA64.dll:12.6.65 
或者直接nvidia-smi (用不了就配置环境变量)

[cuda官网](https://developer.nvidia.com/cuda-toolkit-archive)

选择12.6.3，linux-x86_64-wsl-Ubuntu-2.0-deb(network)
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

配置环境变量
```shell
echo 'export PATH=/usr/local/cuda-12.6/bin:${PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
source ~/.bashrc
nvcc --version # 查看是否成功
```

## 安装cudnn
`lsb_release -a` 或者 `cat /etc/os-release` 查看ubuntu版本

[cudnn官网](https://developer.nvidia.com/cudnn-archive)   
[cuda和cudnn兼容信息](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)   
[计算能力](https://developer.nvidia.com/cuda-gpus)  

选择9.6，linux-x86_64-Ubuntu-24.04-deb(local)  (network好像没有)
```shell
wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2404-9.6.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.6.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

## 设置ncu和nsys

[nvidia官方提示](https://www.nvidia.com/content/control-panel-help/vlatest/zh-cn/mergedprojects/nvdevchs/To_enable_access_to_GPU_performance_counters_for_developrs.htm)

windows -> nivdia控制面板-> 桌面->启用开发者设置-> 开发者：管理GPU性能计数器：允许所有用户访问 GPU 性能计数器 （重启电脑）

nsys 打开： `nsight-sys vect_add.nsys-rep`

## 调试

安装Nsight Visual Studio Code Edition 插件 ,配置launch.json,然后打断点，按F5即可调试,右下角CUDA:(0,0,0)(0,14,0)进行切换比如输入thread(5,0,0)
```shell
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/build/GEMM/profile_cuda_gemm_fp32",
        }, 
        {
            "name" : "CUDA C++: Attach",
            "type" : "cuda-gdb",
            "request" : "attach",
        }
    ]
}
```