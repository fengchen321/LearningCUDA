#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "cuda_gemm.hpp"
#include "profile_utils.cuh"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m <size>    Matrix A rows (default: 1024)" << std::endl;
    std::cout << "  -k <size>    Matrix A cols / Matrix B rows (default: 512)" << std::endl;
    std::cout << "  -n <size>    Matrix B cols (default: 1024)" << std::endl;
    std::cout << "  -r <count>   Number of repeats (default: 10)" << std::endl;
    std::cout << "  -w <count>   Number of warmups (default: 3)" << std::endl;
    std::cout << "  -h           Show this help message" << std::endl;
}

int main(int argc, char* argv[])
{
    // Default values
    size_t m = 1024;
    size_t k = 512;
    size_t n = 1024;
    size_t num_repeats = 10;
    size_t num_warmups = 3;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" && i + 1 < argc) {
            m = std::stoul(argv[++i]);
        } else if (arg == "-k" && i + 1 < argc) {
            k = std::stoul(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            n = std::stoul(argv[++i]);
        } else if (arg == "-r" && i + 1 < argc) {
            num_repeats = std::stoul(argv[++i]);
        } else if (arg == "-w" && i + 1 < argc) {
            num_warmups = std::stoul(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    print_device_info();

    float const fp32_abs_tol{1.0e-3f};
    double const fp32_rel_tol{0.0e-4f};

    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Matrix Size: M = " << m << " N = " << n << " K = " << k << std::endl;
    std::cout << "  Matrix A: " << m << " x " << k << std::endl;
    std::cout << "  Matrix B: " << k << " x " << n << std::endl;
    std::cout << "  Matrix C: " << m << " x " << n << std::endl;
    std::cout << "  Repeats: " << num_repeats << std::endl;
    std::cout << "  Warmups: " << num_warmups << std::endl;
    std::cout << std::endl;

    // Define all the GEMM kernel launch functions to be profiled.
    std::vector<std::pair<
        std::string,
        std::function<void(const Matrix<float>, const Matrix<float>, Matrix<float>, cudaStream_t)>>> const
        gemm_kernel_launch_functions{
            {"Custom GEMM Kernel V00", launch_gemm_kernel_v00<float>},
            {"Custom GEMM Kernel V01", launch_gemm_kernel_v01<float>},
        };

    for (auto const& gemm_kernel_launch_function : gemm_kernel_launch_functions)
    {
        std::cout << gemm_kernel_launch_function.first << std::endl;
        std::pair<float, float> const gemm_kernel_profile_result{
            profile_gemm<float>(
                m, k, n, gemm_kernel_launch_function.second,
                fp32_abs_tol, fp32_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }

    return 0;
}