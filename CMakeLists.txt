cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set CUDA compiler before project() if not already set
# Try to find nvcc in common locations
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    # List of common CUDA installation paths
    set(CUDA_SEARCH_PATHS
        /usr/local/cuda-13.0/bin/nvcc
        /usr/local/cuda-12.8/bin/nvcc
        /usr/local/cuda/bin/nvcc
        /usr/bin/nvcc
    )

    foreach(NVCC_PATH ${CUDA_SEARCH_PATHS})
        if(EXISTS ${NVCC_PATH})
            set(CMAKE_CUDA_COMPILER ${NVCC_PATH})
            message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
            break()
        endif()
    endforeach()

    # If still not found, try to use 'which' to find nvcc in PATH
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        find_program(NVCC_PATH nvcc)
        if(NVCC_PATH)
            set(CMAKE_CUDA_COMPILER ${NVCC_PATH})
            message(STATUS "Found CUDA compiler in PATH: ${CMAKE_CUDA_COMPILER}")
        else()
            message(WARNING "Could not find CUDA compiler (nvcc). Please set CMAKE_CUDA_COMPILER or CUDACXX environment variable.")
        endif()
    endif()
endif()

# Set default CUDA architectures before project() if not already set
# This prevents CMake error: "CMAKE_CUDA_ARCHITECTURES must be non-empty if set"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # Try to detect GPU compute capability early
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE DETECTED_COMPUTE_CAP
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(DETECTED_COMPUTE_CAP)
        # Remove dot: "8.9" -> "89", "12.1" -> "121"
        string(REPLACE "." "" DETECTED_ARCH ${DETECTED_COMPUTE_CAP})

        # For Blackwell (SM121+), map to SM100
        # Blackwell GB10 reports compute capability 12.1, but uses sm_100
        if(DETECTED_ARCH GREATER_EQUAL 121)
            set(CMAKE_CUDA_ARCHITECTURES "100")
            message(STATUS "Auto-detected CUDA architecture: 100 (Blackwell, compute capability ${DETECTED_COMPUTE_CAP})")
        else()
            set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCH})
            message(STATUS "Auto-detected CUDA architecture: ${DETECTED_ARCH} (compute capability ${DETECTED_COMPUTE_CAP})")
        endif()
    else()
        # Default to common architecture if detection fails
        set(CMAKE_CUDA_ARCHITECTURES "89")
        message(STATUS "Could not detect GPU, using default CUDA architecture: 89")
    endif()
endif()

project(explore_gemm CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -Xcompiler=-fno-gnu-unique")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")

# Set up paths for dependencies
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third-party/libtorch)
list(APPEND CMAKE_PREFIX_PATH /usr/local/cuda-12.8)

# PyTorch uses TORCH_CUDA_ARCH_LIST instead of CMAKE_CUDA_ARCHITECTURES
# Set it before find_package(Torch) to avoid incorrect gencode flags
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    # Convert CMAKE_CUDA_ARCHITECTURES format to PyTorch format
    # CMAKE: "100" -> PyTorch: "10.0"
    # CMAKE: "89" -> PyTorch: "8.9"
    # CMAKE: "90a" -> PyTorch: "9.0a"
    set(ARCH_STRING "${CMAKE_CUDA_ARCHITECTURES}")

    if(ARCH_STRING MATCHES "90a")
        set(TORCH_CUDA_ARCH_LIST "9.0")
    elseif(ARCH_STRING STREQUAL "100")
        # For Blackwell GB10 (compute capability 12.1 -> SM100)
        # PyTorch uses "12.1" format for TORCH_CUDA_ARCH_LIST
        set(TORCH_CUDA_ARCH_LIST "12.1")
        message(STATUS "Blackwell detected: Using TORCH_CUDA_ARCH_LIST=12.1")
    else()
        # Convert from "89" to "8.9" format
        string(LENGTH "${ARCH_STRING}" ARCH_LEN)
        if(ARCH_LEN EQUAL 2)
            string(SUBSTRING "${ARCH_STRING}" 0 1 MAJOR)
            string(SUBSTRING "${ARCH_STRING}" 1 1 MINOR)
            set(TORCH_CUDA_ARCH_LIST "${MAJOR}.${MINOR}")
        elseif(ARCH_LEN EQUAL 3)
            string(SUBSTRING "${ARCH_STRING}" 0 2 MAJOR)
            string(SUBSTRING "${ARCH_STRING}" 2 1 MINOR)
            set(TORCH_CUDA_ARCH_LIST "${MAJOR}.${MINOR}")
        else()
            set(TORCH_CUDA_ARCH_LIST "${ARCH_STRING}")
        endif()
    endif()
    message(STATUS "Setting TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
endif()

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

# Detect GPU compute capability
# Allow manual override via -DBUILD_HOPPER_KERNELS=ON
if(NOT DEFINED BUILD_HOPPER_KERNELS)
    set(GPU_COMPUTE_CAPABILITY 0)

    # Use nvidia-smi to detect installed GPU
    if(CMAKE_CUDA_COMPILER)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE GPU_COMPUTE_CAP_OUTPUT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(GPU_COMPUTE_CAP_OUTPUT)
            # Remove the dot to convert "8.9" to "89"
            string(REPLACE "." "" GPU_COMPUTE_CAPABILITY ${GPU_COMPUTE_CAP_OUTPUT})
            message(STATUS "Detected GPU compute capability from nvidia-smi: SM${GPU_COMPUTE_CAPABILITY}")
        else()
            message(WARNING "Could not detect GPU compute capability. Assuming SM89 (Ada Lovelace).")
            set(GPU_COMPUTE_CAPABILITY 89)
        endif()
    endif()

    # Determine if we should build Hopper kernels
    # Note: Only Hopper (SM90-SM99) supports Hopper-specific features (TMA, GMMA)
    # Blackwell (SM121+) is a different architecture and doesn't support Hopper instructions
    set(BUILD_HOPPER_KERNELS OFF)
    if(GPU_COMPUTE_CAPABILITY GREATER_EQUAL 90 AND GPU_COMPUTE_CAPABILITY LESS 100)
        set(BUILD_HOPPER_KERNELS ON)
        message(STATUS "GPU supports Hopper (SM90-SM99): Hopper kernels will be built")
    elseif(GPU_COMPUTE_CAPABILITY GREATER_EQUAL 121)
        message(STATUS "GPU is Blackwell (SM${GPU_COMPUTE_CAPABILITY}): Skipping Hopper kernels (Hopper-specific features not available)")
    else()
        message(STATUS "GPU compute capability SM${GPU_COMPUTE_CAPABILITY} < SM90: Skipping Hopper kernels")
    endif()
else()
    # User manually specified BUILD_HOPPER_KERNELS
    if(BUILD_HOPPER_KERNELS)
        message(STATUS "BUILD_HOPPER_KERNELS manually enabled by user")
    else()
        message(STATUS "BUILD_HOPPER_KERNELS manually disabled by user")
    endif()
endif()

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

    # CUTLASS Hopper kernel (requires SM90a+ and CUTLASS 3.x)
    if(BUILD_HOPPER_KERNELS)
        message(STATUS "Building CUTLASS Hopper kernel for SM90a")
        add_executable(test_gemm_cutlass_hopper
            ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_cutlass_hopper.cpp
            ${CMAKE_SOURCE_DIR}/cuda/15_kernel_cutlass_hopper.cu)
        target_link_libraries(test_gemm_cutlass_hopper "${TORCH_LIBRARIES}")
        # Hopper GMMA instructions require sm_90a
        set_property(TARGET test_gemm_cutlass_hopper PROPERTY CUDA_ARCHITECTURES 90a)

        # Hopper Benchmark (requires SM90a+ and CUTLASS 3.x)
        message(STATUS "Building Hopper benchmark for SM90a")
        add_executable(benchmark_hopper
            ${CMAKE_SOURCE_DIR}/cuda/scripts/benchmark_hopper.cu)
        target_link_libraries(benchmark_hopper "${TORCH_LIBRARIES}")
        # Hopper GMMA instructions require sm_90a
        set_property(TARGET benchmark_hopper PROPERTY CUDA_ARCHITECTURES 90a)
    else()
        message(STATUS "Skipping CUTLASS Hopper kernel (requires SM90+)")
    endif()

    # Detect GPU compute capability for Blackwell
    # Blackwell requires SM100+ (compute capability 10.0+)
    set(BUILD_BLACKWELL_KERNELS OFF)
    if(GPU_COMPUTE_CAPABILITY GREATER_EQUAL 100)
        set(BUILD_BLACKWELL_KERNELS ON)
        message(STATUS "GPU supports Blackwell (SM100+): Blackwell kernels will be built")
    else()
        message(STATUS "GPU compute capability SM${GPU_COMPUTE_CAPABILITY} < SM100: Skipping Blackwell kernels")
    endif()

    # CUTLASS Blackwell kernel (requires SM100+ and CUTLASS 3.x)
    if(BUILD_BLACKWELL_KERNELS)
        message(STATUS "Building CUTLASS Blackwell kernel for SM100")
        add_executable(test_gemm_cutlass_blackwell
            ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_cutlass_blackwell.cpp
            ${CMAKE_SOURCE_DIR}/cuda/17_kernel_cutlass_blackwell.cu)
        target_link_libraries(test_gemm_cutlass_blackwell "${TORCH_LIBRARIES}")
        # Blackwell requires sm_100
        set_property(TARGET test_gemm_cutlass_blackwell PROPERTY CUDA_ARCHITECTURES 100)

        # Blackwell Benchmark (requires SM100+ and CUTLASS 3.x)
        message(STATUS "Building Blackwell benchmark for SM100")
        add_executable(benchmark_blackwell
            ${CMAKE_SOURCE_DIR}/cuda/scripts/benchmark_blackwell.cu)
        target_link_libraries(benchmark_blackwell "${TORCH_LIBRARIES}")
        # Blackwell requires sm_100
        set_property(TARGET benchmark_blackwell PROPERTY CUDA_ARCHITECTURES 100)
    else()
        message(STATUS "Skipping CUTLASS Blackwell kernel (requires SM100+)")
    endif()
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

# Only add Hopper test if it was built
if(BUILD_HOPPER_KERNELS)
    add_test(NAME test_gemm_cutlass_hopper COMMAND test_gemm_cutlass_hopper)
endif()

# Only add Blackwell test if it was built
if(BUILD_BLACKWELL_KERNELS)
    add_test(NAME test_gemm_cutlass_blackwell COMMAND test_gemm_cutlass_blackwell)
endif()


# NOTES
# Build with: cmake -B build
#
# Auto-detection:
#   - CUDA architecture is automatically detected from installed GPU using nvidia-smi
#   - Falls back to SM89 if detection fails
#   - Hopper kernels (SM90+) are conditionally compiled based on detected architecture
#
# Override CUDA architecture:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a  # For Hopper
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100  # For Blackwell (GB10, etc.)
#
# Manually enable/disable Hopper kernels:
#   cmake -B build -DBUILD_HOPPER_KERNELS=ON -DCMAKE_CUDA_ARCHITECTURES=90a
#   cmake -B build -DBUILD_HOPPER_KERNELS=OFF
#
# Specify CUDA compiler:
#   cmake -B build -DCMAKE_CUDA_COMPILER=/path/to/nvcc
#
# Then: cmake --build build
# Run tests: ctest --test-dir build
#
# Note: Hopper kernels (SM90+) are automatically detected and built only if:
#   - GPU compute capability >= SM90 (detected via nvidia-smi), OR
#   - BUILD_HOPPER_KERNELS is manually set to ON