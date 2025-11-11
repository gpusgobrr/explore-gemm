cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set CUDA compiler before project() if not already set
# INSTALL CUDA 12.8 on Debian 12 with:
# wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-12-8
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS /usr/local/cuda-12.8/bin/nvcc)
        set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
        message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    endif()
endif()

project(explore_gemm CUDA CXX)

# Set CUDA architecture if not already set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -Xcompiler=-fno-gnu-unique")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")

# Set up paths for dependencies
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third-party/libtorch)
list(APPEND CMAKE_PREFIX_PATH /usr/local/cuda-12.8)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third-party)

# CUTLASS library configuration
# Try to find CUTLASS in multiple locations
set(CUTLASS_AVAILABLE FALSE)

# Location 1: Local third-party directory (preferred - should be CUTLASS 3.x)
if(EXISTS ${CMAKE_SOURCE_DIR}/third-party/cutlass/include)
    set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third-party/cutlass/include)
    set(CUTLASS_UTILS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third-party/cutlass/tools/util/include)
    set(CUTLASS_AVAILABLE TRUE)
    message(STATUS "CUTLASS found at ${CUTLASS_INCLUDE_DIR} (using local installation)")
endif()

if(CUTLASS_AVAILABLE)
    include_directories(${CUTLASS_INCLUDE_DIR})
    include_directories(${CUTLASS_UTILS_INCLUDE_DIR})
endif()

# Tests - each test now only compiles its specific CUDA kernel file
add_executable(test_gemm_naive
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_naive.cpp
    ${CMAKE_SOURCE_DIR}/cuda/01_naive.cu)
target_link_libraries(test_gemm_naive "${TORCH_LIBRARIES}")

add_executable(test_gemm_coalesce
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_coalesce.cpp
    ${CMAKE_SOURCE_DIR}/cuda/02_kernel_global_mem_coalesce.cu
    ${CMAKE_SOURCE_DIR}/cuda/01_naive.cu)  # Needed for comparison test
target_link_libraries(test_gemm_coalesce "${TORCH_LIBRARIES}")

add_executable(test_gemm_shared_mem
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_shared_mem.cpp
    ${CMAKE_SOURCE_DIR}/cuda/03_kernel_shared_mem.cu)
target_link_libraries(test_gemm_shared_mem "${TORCH_LIBRARIES}")

add_executable(test_gemm_blocktiling1d
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_blocktiling1d.cpp
    ${CMAKE_SOURCE_DIR}/cuda/04_kernel_blocktiling_1d.cu)
target_link_libraries(test_gemm_blocktiling1d "${TORCH_LIBRARIES}")

add_executable(test_gemm_blocktiling2d
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_blocktiling2d.cpp
    ${CMAKE_SOURCE_DIR}/cuda/05_kernel_blocktiling_2d.cu)
target_link_libraries(test_gemm_blocktiling2d "${TORCH_LIBRARIES}")

add_executable(test_gemm_vectorize
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_vectorize.cpp
    ${CMAKE_SOURCE_DIR}/cuda/06_kernel_vectorize.cu)
target_link_libraries(test_gemm_vectorize "${TORCH_LIBRARIES}")

add_executable(test_gemm_warptiling
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_warptiling.cpp
    ${CMAKE_SOURCE_DIR}/cuda/07_kernel_warptiling.cu)
target_link_libraries(test_gemm_warptiling "${TORCH_LIBRARIES}")

add_executable(test_gemm_warptiling_all_dtypes
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_warptiling_all_dtypes.cpp
    ${CMAKE_SOURCE_DIR}/cuda/08_kernel_warptiling_all_dtypes.cu
    ${CMAKE_SOURCE_DIR}/cuda/07_kernel_warptiling.cu)  # Needed for sgemm_warptiling_default
target_link_libraries(test_gemm_warptiling_all_dtypes "${TORCH_LIBRARIES}")

add_executable(test_gemm_tensorcore_naive
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_tensorcore_naive.cpp
    ${CMAKE_SOURCE_DIR}/cuda/09_kernel_tensorcore_naive.cu)
target_link_libraries(test_gemm_tensorcore_naive "${TORCH_LIBRARIES}")

add_executable(test_gemm_tensorcore
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_tensorcore.cpp
    ${CMAKE_SOURCE_DIR}/cuda/10_kernel_tensorcore_warptiled.cu)
target_link_libraries(test_gemm_tensorcore "${TORCH_LIBRARIES}")

add_executable(test_gemm_tensorcore_double_buffered
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_tensorcore_double_buffered.cpp
    ${CMAKE_SOURCE_DIR}/cuda/11_kernel_tensorcore_double_buffered.cu)
target_link_libraries(test_gemm_tensorcore_double_buffered "${TORCH_LIBRARIES}")

add_executable(test_gemm_tensorcore_async
    ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_tensorcore_async.cpp
    ${CMAKE_SOURCE_DIR}/cuda/12_kernel_tensorcore_async.cu
    ${CMAKE_SOURCE_DIR}/cuda/11_kernel_tensorcore_double_buffered.cu)  # Needed for comparison test
target_link_libraries(test_gemm_tensorcore_async "${TORCH_LIBRARIES}")

# CUTLASS test (only build if CUTLASS is available)
if(CUTLASS_AVAILABLE)
    add_executable(test_gemm_cutlass
        ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_cutlass.cpp
        ${CMAKE_SOURCE_DIR}/cuda/13_kernel_cutlass.cu)
    target_link_libraries(test_gemm_cutlass "${TORCH_LIBRARIES}")

    # CUTLASS autotunable kernel - used by Python autotuner
    # This compiles all configuration variants for autotuning
    add_library(cutlass_autotunable SHARED
        ${CMAKE_SOURCE_DIR}/cuda/14_kernel_cutlass_autotunable.cu)
    target_link_libraries(cutlass_autotunable "${TORCH_LIBRARIES}")
    set_target_properties(cutlass_autotunable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

    # FP4 GEMV example (CUTLASS example from scripts directory)
    add_executable(fp4_gemv_example
        ${CMAKE_SOURCE_DIR}/cuda/scripts/fp4_gemv.cu)
    target_link_libraries(fp4_gemv_example "${TORCH_LIBRARIES}")

    # Batched GEMM example (CUTLASS example from scripts directory)
    add_executable(batched_gemm_example
        ${CMAKE_SOURCE_DIR}/cuda/scripts/batched_gemm.cu)
    target_link_libraries(batched_gemm_example "${TORCH_LIBRARIES}")

    # Batched FP4 GEMV example (combined pattern from batched GEMM + FP4 GEMV)
    add_executable(batched_fp4_gemv_example
        ${CMAKE_SOURCE_DIR}/cuda/scripts/batched_fp4_gemv.cu)
    target_link_libraries(batched_fp4_gemv_example "${TORCH_LIBRARIES}")

    # Batched FP4 GEMV executable (accepts PyTorch tensors as inputs)
    add_executable(batched_fp4_gemv_pytorch
        ${CMAKE_SOURCE_DIR}/cuda/scripts/batched_fp4_gemv_pytorch.cu)
    target_link_libraries(batched_fp4_gemv_pytorch "${TORCH_LIBRARIES}")
endif()

enable_testing()

add_test(NAME test_gemm_naive COMMAND test_gemm_naive)
add_test(NAME test_gemm_coalesce COMMAND test_gemm_coalesce)
add_test(NAME test_gemm_shared_mem COMMAND test_gemm_shared_mem)
add_test(NAME test_gemm_blocktiling1d COMMAND test_gemm_blocktiling1d)
add_test(NAME test_gemm_blocktiling2d COMMAND test_gemm_blocktiling2d)
add_test(NAME test_gemm_vectorize COMMAND test_gemm_vectorize)
add_test(NAME test_gemm_warptiling COMMAND test_gemm_warptiling)
add_test(NAME test_gemm_warptiling_all_dtypes COMMAND test_gemm_warptiling_all_dtypes)
add_test(NAME test_gemm_tensorcore_naive COMMAND test_gemm_tensorcore_naive)
add_test(NAME test_gemm_tensorcore COMMAND test_gemm_tensorcore)
add_test(NAME test_gemm_tensorcore_double_buffered COMMAND test_gemm_tensorcore_double_buffered)
add_test(NAME test_gemm_tensorcore_async COMMAND test_gemm_tensorcore_async)

if(CUTLASS_AVAILABLE)
    add_test(NAME test_gemm_cutlass COMMAND test_gemm_cutlass)
endif()

# NOTES
# Build with: cmake -B build
# Or override defaults: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/path/to/nvcc
# Then: cmake --build build
# Run tests: ctest --test-dir build