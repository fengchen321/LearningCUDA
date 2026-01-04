#ifndef PROFILE_UTILS_CUH
#define PROFILE_UTILS_CUH

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

template <typename T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 100,
                          size_t num_warmups = 100) {
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

#define CHECK_CUBALS_ERROR(val) check_cublas((val), #val, __FILE__, __LINE__)
void check_cublas(cublasStatus_t err, const char* const func, const char* const file, const int line) {
    (void)func;  // Suppress unused parameter warning
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error at: " << file << ": " << line << "\n";
        std::cerr <<cublasGetStatusString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// Determine CUDA data type from type
template <typename T,
            typename std::enable_if<std::is_same<T, float>::value ||
                                        std::is_same<T, double>::value,
                                        bool>::type = true>
constexpr cudaDataType_t cuda_data_type_trait() {
    if (std::is_same<T, float>::value) return CUDA_R_32F;
    else if (std::is_same<T, double>::value) return CUDA_R_64F;
    else {
        throw std::runtime_error("Unsupported data type.");
    }
}

template <typename T,
            typename std::enable_if<std::is_same<T, float>::value ||
                                        std::is_same<T, double>::value,
                                        bool>::type = true>
void launch_gemm_cublas(const Matrix<T> A, const Matrix<T> B, Matrix<T> C, cublasHandle_t handle) {
    constexpr cublasGemmAlgo_t algo{CUBLAS_GEMM_DEFAULT};
    constexpr cudaDataType_t data_type{cuda_data_type_trait<T>()};
    T const alpha{static_cast<T>(1.0)};
    T const beta{static_cast<T>(0.0)};

    CHECK_CUBALS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B.width, A.height, A.width, &alpha, 
                            B.elements, data_type, B.width, 
                            A.elements, data_type, A.width, &beta, 
                            C.elements, data_type, C.width, data_type, algo));
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value,
                                  bool>::type = true>
void launch_gemm_cpu(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
    // Compute GEMM using CPU.
    for (size_t i = 0; i < A.height; ++i) {
        for (size_t j = 0; j < B.width; ++j) {
            T sum{static_cast<T>(0)};
            for (size_t k = 0; k < A.width; ++k) {
                sum += A.elements[i * A.width + k] * B.elements[k * B.width + j];
            }
            C.elements[i * C.width + j] = sum;
        }
    }
}

template <typename T>
bool all_close(const Matrix<T> A, const Matrix<T> B,
               T abs_tol, double rel_tol) {
    for (size_t i = 0; i < A.height; ++i) {
        for (size_t j = 0; j < A.width; ++j) {
            double const a = static_cast<double>(A.elements[i * A.width + j]);
            double const b = static_cast<double>(B.elements[i * B.width + j]);
            double const diff = a - b;
            double const diff_val = std::abs(diff);
            if (diff_val >
                std::max(static_cast<double>(abs_tol),
                         static_cast<double>(std::abs(b)) * rel_tol)) {
                std::cout << "A[" << i << ", " << j << "] = " << a
                          << " B[" << i << ", " << j << "] = " << b
                          << " Abs Diff: " << diff_val
                          << " Abs Diff Threshold: "
                          << static_cast<double>(abs_tol)
                          << " Rel->Abs Diff Threshold: "
                          << static_cast<double>(
                                 static_cast<double>(std::abs(b)) *
                                 rel_tol)
                          << std::endl;
                return false;
            }
        }
    }
    return true;
}

void print_device_info() {
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13
    int memClockKhz = 0;
    cudaDeviceGetAttribute(&memClockKhz, cudaDevAttrMemoryClockRate, device_id);
#else
    int memClockKhz = device_prop.memoryClockRate;
#endif
    float const peak_bandwidth{
        static_cast<float>(2.0f * memClockKhz *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << std::endl;
}

template <typename T>
float compute_effective_bandwidth(const Matrix<T> A, const Matrix<T> B, float latency)
{
    assert(A.width==B.height);
    return ((A.height * A.width + B.height * B.width + A.height * B.width) * sizeof(T)) / (latency * 1e-3) / 1e9;
}

template <typename T>
float compute_effective_tflops(const Matrix<T> A, const Matrix<T> B, float latency)
{
    return (2.0 * A.height * A.width * B.width) / (latency * 1e-3) / 1e12;
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value,
                                  bool>::type = true>
void random_initialize_matrix(Matrix<T> M, unsigned int seed = 0U)
{
    std::default_random_engine eng(seed);
    // The best way to verify is to use integer values.
    std::uniform_int_distribution<int> dis(0, 5);
    // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    for (size_t i = 0; i < M.height; ++i) {
        for (size_t j = 0; j < M.width; ++j) {
            M.elements[i * M.width + j] = static_cast<T>(rand());
        }
    }
}

template <typename T>
void print_performance_result(const Matrix<T> A, const Matrix<T> B, float latency)
{
    float const effective_bandwidth{
        compute_effective_bandwidth<T>(A, B, latency)};
    float const effective_tflops{compute_effective_tflops<T>(A, B, latency)};

    std::cout << "Latency: " << latency << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth << " GB/s"
              << std::endl;
    std::cout << "Effective TFLOPS: " << effective_tflops << " TFLOPS"
              << std::endl;
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value,
                                  bool>::type = true>
std::pair<float, float> profile_gemm(
    int m, int k, int n,
    std::function<void(const Matrix<T>, const Matrix<T>, Matrix<T>, cudaStream_t)>
        gemm_kernel_launch_function,
    T abs_tol, double rel_tol, size_t num_repeats = 10, size_t num_warmups = 10,
    unsigned int seed = 0U)
{
    (void)seed;  // Suppress unused parameter warning
    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Allocate memory on host.
    Matrix<T> A_host, B_host, C_host, C_host_from_device, C_host_ref;

    A_host.height = m; A_host.width = k;
    B_host.height = k; B_host.width = n;
    C_host.height =  C_host_from_device.height = C_host_ref.height = m; 
    C_host.width = C_host_from_device.width = C_host_ref.width = n;
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host.elements, A_host.width * A_host.height * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host.elements, B_host.width * B_host.height  * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host.elements, C_host.width * C_host.height  * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_from_device.elements, C_host_from_device.width * C_host_from_device.height * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_ref.elements, C_host_ref.width * C_host_ref.height * sizeof(T)));

    // Initialize matrix A and B.
    random_initialize_matrix(A_host);
    random_initialize_matrix(B_host);
    random_initialize_matrix(C_host);

    // Allocate memory on device.
    Matrix<T> A_device, B_device, C_device;
    A_device.height = A_host.height; A_device.width = A_device.stride = A_host.width ;
    B_device.height = B_host.height; B_device.width = B_device.stride = B_host.width ;
    C_device.height = C_host.height; C_device.width = C_device.stride = C_host.width ;
    CHECK_CUDA_ERROR(cudaMalloc(&A_device.elements, A_device.width * A_device.height * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device.elements, B_device.width * B_device.height * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device.elements, C_device.width * C_device.height * sizeof(T)));

    // Copy matrix A and B from host to device.
    CHECK_CUDA_ERROR(cudaMemcpy(A_device.elements, A_host.elements, 
                                A_device.width * A_device.height * sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device.elements, B_host.elements, 
                                B_device.width * B_device.height* sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_device.elements, C_host.elements, 
                                C_device.width * C_device.height * sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_ref.elements, C_host.elements, 
                                C_host_ref.width * C_host_ref.height * sizeof(T),
                                cudaMemcpyHostToHost));
    
    // Create cuBlas handle.
    cublasHandle_t handle;
    CHECK_CUBALS_ERROR(cublasCreate(&handle));
    CHECK_CUBALS_ERROR(cublasSetStream(handle, stream));

    // Compute reference output using cuBLAS.
    launch_gemm_cublas<T>(A_device, B_device, C_device, handle);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_ref.elements, C_device.elements,
                                C_host_ref.width * C_host_ref.height * sizeof(T), cudaMemcpyDeviceToHost));

    // // Compute reference output using CPU.
    // std::cout << "Computing reference output using CPU..." << std::endl;
    // launch_gemm_cpu(A_host, B_host, C_host);
    // std::cout << "Done." << std::endl;

    // Launch CUDA GEMM.
    CHECK_CUDA_ERROR(cudaMemcpy(C_device.elements, C_host.elements,
                                C_device.width * C_device.height * sizeof(T), cudaMemcpyHostToDevice));
    // Verify the correctness of CUDA GEMM.
    gemm_kernel_launch_function(A_device, B_device, C_device, stream);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device.elements, C_device.elements,
                                C_host_from_device.width * C_host_from_device.height * sizeof(T), cudaMemcpyDeviceToHost));
    assert(all_close<T>(C_host_from_device, C_host_ref, abs_tol, rel_tol));

    float const latency_cublas{measure_performance<void>(
        [&](cudaStream_t stream)
        {
            (void)stream;  // Suppress unused parameter warning
            launch_gemm_cublas<T>(A_device, B_device, C_device, handle);
            return;
        },
        stream, num_repeats, num_warmups)};

    float const latency_cuda_gemm{measure_performance<void>(
        [&](cudaStream_t stream)
        {
            (void)stream;  // Suppress unused parameter warning
            gemm_kernel_launch_function(A_device, B_device, C_device, stream);
            return;
        },
        stream, num_repeats, num_warmups)};

    // Release resources.
    CHECK_CUDA_ERROR(cudaFree(A_device.elements));
    CHECK_CUDA_ERROR(cudaFree(B_device.elements));
    CHECK_CUDA_ERROR(cudaFree(C_device.elements));
    CHECK_CUDA_ERROR(cudaFreeHost(A_host.elements));
    CHECK_CUDA_ERROR(cudaFreeHost(B_host.elements));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host.elements));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_from_device.elements));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    std::cout << "cuBLAS GEMM Kernel Performance" << std::endl;
    print_performance_result<T>(A_host, B_host, latency_cublas);
    std::cout << "Custom GEMM Kernel Performance" << std::endl;
    print_performance_result<T>(A_host, B_host, latency_cuda_gemm);
    std::cout << "Custom GEMM VS cuBLAS GEMM Performance: "
              << latency_cublas / latency_cuda_gemm * 100.0f << "%\n";

    return std::pair<float, float>{latency_cublas, latency_cuda_gemm};
}

#endif // PROFILE_UTILS_CUH