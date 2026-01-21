cmake_minimum_required(VERSION 3.18)
project(profile_demo LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find CUDA
find_package(CUDA REQUIRED)

# Public include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Demo 1: Basic vector add
add_executable(basic 01_basic/src/basic.cu)
target_link_libraries(basic ${CUDA_LIBRARIES})

# Demo 2: Memory transfer test
add_executable(memory_test 02_memory/src/memory_test.cu)
target_link_libraries(memory_test ${CUDA_LIBRARIES})

# Demo 3: Multi-stream test
add_executable(multistream 03_multistream/src/multistream.cu)
target_link_libraries(multistream ${CUDA_LIBRARIES})

# Demo 4: NVTX marking test (optional, requires NVTX v3 headers)
# NVTX v3 is header-only, provided by CUDA Toolkit or NVIDIA tools
# https://nvidia.github.io/NVTX/
set(NVTX_PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    /usr/local/cuda/include
)
find_path(NVTX_INCLUDE_DIRS NAMES nvtx3/nvToolsExt.h PATHS ${NVTX_PATHS})
if(NVTX_INCLUDE_DIRS)
    message(STATUS "NVTX v3 found: ${NVTX_INCLUDE_DIRS}")
    include_directories(${NVTX_INCLUDE_DIRS})
    add_executable(nvtx_demo 04_nvtx/src/nvtx_demo.cu)
    target_link_libraries(nvtx_demo ${CUDA_LIBRARIES})
else()
    message(STATUS "NVTX v3 not found, skipping nvtx_demo")
endif()

# Demo 5: Unified memory test
add_executable(um_demo 05_unified_memory/src/um_demo.cu)
target_link_libraries(um_demo ${CUDA_LIBRARIES})

# Demo 6: CUDA Graphs test
add_executable(cuda_graph 06_cuda_graph/src/cuda_graph.cu)
target_link_libraries(cuda_graph ${CUDA_LIBRARIES})

# Demo 7: Callstack test
add_executable(callstack 07_callstack/src/callstack.cu)
target_link_libraries(callstack ${CUDA_LIBRARIES})

# Demo 8: fork-exec test
add_executable(fork_exec 08_fork_exec/src/fork_exec.cu)
target_link_libraries(fork_exec ${CUDA_LIBRARIES})

# Demo 9: MPI test (optional, requires MPI)
find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI found, building mpi_demo")
    cuda_add_executable(mpi_demo 09_mpi/src/mpi_demo.cu)
    target_link_libraries(mpi_demo ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
else()
    message(STATUS "MPI not found, skipping mpi_demo")
endif()

# Demo 10: CUPTI test
# CUPTI is part of CUDA Toolkit, find it via CUDA paths
set(CUPTI_PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
    /usr/local/cuda/lib64
    /usr/local/cuda/lib
)
find_library(CUPTI_LIBRARY NAMES cupti PATHS ${CUPTI_PATHS})
if(CUPTI_LIBRARY)
    message(STATUS "CUPTI found: ${CUPTI_LIBRARY}")
    cuda_add_executable(activity_trace_async 10_cupti/src/mat.cu 10_cupti/src/activity_trace_async.cpp)
    target_link_libraries(activity_trace_async ${CUDA_LIBRARIES} ${CUPTI_LIBRARY})
else()
    message(STATUS "CUPTI not found, skipping activity_trace_async")
endif()
