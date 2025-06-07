#include <cuda_runtime.h>
#include "cuda_gemm.hpp"
#include "profile_utils.cuh"

int main()
{
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{1U};

    float const fp32_abs_tol{1.0e-3f};
    double const fp32_rel_tol{0.0e-4f};

    constexpr size_t m{1024};
    constexpr size_t k{512};
    constexpr size_t n{1024};

    std::cout << "Matrix Size: " << "M = " << m << " N = " << n << " K = " << k
              << std::endl;
    std::cout << "Matrix A: " << m << " x " << k << std::endl;
    std::cout << "Matrix B: " << k << " x " << n << std::endl;
    std::cout << "Matrix C: " << m << " x " << n << std::endl;
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