"""Shared CUDA extension loader for GEMM kernels.

This module provides a centralized way to build and load CUDA GEMM kernels
for both benchmark.py and kernel_runner.py, ensuring consistent build configuration.
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load_inline

# Try to import loguru for nicer logging, fall back to print
try:
    from loguru import logger
except ImportError:
    # Fallback logger if loguru is not installed
    class SimpleLogger:
        def info(self, msg):
            print(f"INFO: {msg}")

        def success(self, msg):
            print(f"SUCCESS: {msg}")

        def warning(self, msg):
            print(f"WARNING: {msg}")

        def error(self, msg):
            print(f"ERROR: {msg}")

    logger = SimpleLogger()


def get_cuda_code(
    cuda_file: str, header_file: str, utils_header_file: str
) -> Tuple[str, str, str]:
    """Load CUDA source and header files, removing #include and #pragma once directives.

    Args:
        cuda_file: Path to the CUDA source file
        header_file: Path to the main header file (gemm_kernels.cuh)
        utils_header_file: Path to utilities header file (utils.cuh)

    Returns:
        Tuple of (cuda_code, header_code, utils_code)
    """
    with open(cuda_file) as f:
        cuda_code = "".join(
            [line for line in f.readlines() if not line.startswith("#include")]
        )

    with open(header_file) as f:
        header_code = "".join(
            [
                line
                for line in f.readlines()
                if not line.startswith("#include")
                and not line.startswith("#pragma once")
            ]
        )

    with open(utils_header_file) as f:
        utils_code = "".join(
            [
                line
                for line in f.readlines()
                if not line.startswith("#include")
                and not line.startswith("#pragma once")
            ]
        )

    return cuda_code, header_code, utils_code


def find_cuda_home():
    """Find CUDA installation directory.

    Searches in common CUDA installation locations and returns the first valid path.
    Also sets the CUDA_HOME environment variable if not already set.

    Returns:
        Path to CUDA installation, or None if not found
    """
    # Check if CUDA_HOME is already set
    if os.environ.get("CUDA_HOME"):
        cuda_home = Path(os.environ["CUDA_HOME"])
        if cuda_home.exists():
            # Ensure CUDA_HOME is explicitly set in the environment if already found and valid
            os.environ["CUDA_HOME"] = str(cuda_home)
            return str(cuda_home)

    # Second, check the default symlink path `/usr/local/cuda`
    default_cuda_path = Path("/usr/local/cuda")
    if (default_cuda_path / "bin" / "nvcc").exists():
        cuda_home = str(default_cuda_path)
        os.environ["CUDA_HOME"] = cuda_home
        return cuda_home

    return None


def create_cuda_extension(
    verbose: bool = True,
    load_autotune_kernels: bool = False,
    load_hopper_kernels: bool = False,
):
    """Create PyTorch extension for CUDA GEMM kernels.

    This function handles all the complexity of building CUDA kernels with
    proper support for Tensor Cores (FP16/BF16 via WMMA API).

    Args:
        verbose: Whether to show verbose build output (default: True)
        load_autotune_kernels: Whether to load autotunable kernel versions (default: False)
        load_hopper_kernels: Whether to load Hopper-specific kernels (default: False)

    Returns:
        The loaded CUDA extension module
    """
    # Find and set CUDA_HOME before building extensions
    # The find_cuda_home function will set the environment variable.
    cuda_home = find_cuda_home()
    if cuda_home:
        logger.info(f"🔧 Using CUDA installation at: {cuda_home}")
    else:
        logger.warning("⚠️  Could not find CUDA installation. Extension build may fail.")

    # Also set CUDACXX to ensure nvcc is found
    if cuda_home:
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        os.environ["CUDACXX"] = nvcc_path

    file_dir = Path(__file__).parent.parent

    # Load all CUDA source files
    naive_cu = file_dir / "01_naive.cu"
    coalesce_cu = file_dir / "02_kernel_global_mem_coalesce.cu"
    shared_mem_cu = file_dir / "03_kernel_shared_mem.cu"
    blocktiling_1d_cu = file_dir / "04_kernel_blocktiling_1d.cu"
    blocktiling_2d_cu = file_dir / "05_kernel_blocktiling_2d.cu"
    vectorize_cu = file_dir / "06_kernel_vectorize.cu"
    warptiling_cu = file_dir / "07_kernel_warptiling.cu"
    warptiling_multidtype_cu = file_dir / "08_kernel_warptiling_all_dtypes.cu"
    tensorcore_naive_cu = file_dir / "09_kernel_tensorcore_naive.cu"
    tensorcore_cu = file_dir / "10_kernel_tensorcore_warptiled.cu"
    tensorcore_double_buffered_cu = file_dir / "11_kernel_tensorcore_double_buffered.cu"
    tensorcore_async_cu = file_dir / "12_kernel_tensorcore_async.cu"
    cutlass_cu = file_dir / "13_kernel_cutlass.cu"
    cutlass_autotune_cu = file_dir / "14_kernel_cutlass_autotunable.cu"
    cutlass_hopper_cu = file_dir / "15_kernel_cutlass_hopper.cu"
    cutlass_hopper_autotune_cu = file_dir / "16_kernel_cutlass_hopper_autotunable.cu"
    header_file = file_dir / "gemm_kernels.cuh"
    utils_header_file = file_dir / "utils.cuh"

    # Check GPU compute capability to determine if we should load Hopper kernels
    import torch

    has_hopper = False
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = device_props.major * 10 + device_props.minor

        # Blackwell (SM120+) should NOT load Hopper kernels
        if compute_capability >= 120:
            has_hopper = False
            if verbose:
                logger.info(f"🖥️  GPU: {device_props.name}")
                logger.info(
                    f"   Compute Capability: SM{device_props.major}.{device_props.minor}"
                )
                logger.info(
                    f"   ℹ️  Blackwell (SM120+) detected - skipping Hopper kernels"
                )
        else:
            has_hopper = (compute_capability >= 90) and load_hopper_kernels

            if verbose:
                logger.info(f"🖥️  GPU: {device_props.name}")
                logger.info(
                    f"   Compute Capability: SM{device_props.major}.{device_props.minor}"
                )
                if has_hopper:
                    logger.info(
                        f"   ✅ Hopper (SM90+) support detected - loading Hopper kernels"
                    )
                elif not load_hopper_kernels:
                    if compute_capability >= 90:
                        logger.info(
                            f"   ℹ️  Hopper kernels explicitly disabled via flag - skipping Hopper kernels"
                        )
                    else:
                        logger.warning(
                            f"   ⚠️  Hopper (SM90+) not available on this GPU - skipping Hopper kernels"
                        )

    else:
        if verbose:
            logger.warning("   ⚠️  CUDA not available - skipping Hopper kernels")

    if verbose:
        logger.info("📂 Loading CUDA sources:")
        logger.info(f"   • Naive: {naive_cu}")
        logger.info(f"   • Coalesced: {coalesce_cu}")
        logger.info(f"   • Shared Memory: {shared_mem_cu}")
        logger.info(f"   • 1D Block Tiling: {blocktiling_1d_cu}")
        logger.info(f"   • 2D Block Tiling: {blocktiling_2d_cu}")
        logger.info(f"   • Vectorize: {vectorize_cu}")
        logger.info(f"   • Warptiling (FP32): {warptiling_cu}")
        logger.info(f"   • Warptiling (Multi-Dtype): {warptiling_multidtype_cu}")
        logger.info(f"   • Tensor Core (Naive): {tensorcore_naive_cu}")
        logger.info(f"   • Tensor Core (Warptiled): {tensorcore_cu}")
        logger.info(
            f"   • Tensor Core Double Buffered: {tensorcore_double_buffered_cu}"
        )
        logger.info(f"   • Tensor Core Async: {tensorcore_async_cu}")
        logger.info(f"   • CUTLASS: {cutlass_cu}")
        if load_autotune_kernels:
            logger.info(f"   • CUTLASS Autotunable: {cutlass_autotune_cu}")
        if has_hopper:
            logger.info(f"   • CUTLASS Hopper: {cutlass_hopper_cu}")
            if load_autotune_kernels:
                logger.info(
                    f"   • CUTLASS Hopper Autotunable: {cutlass_hopper_autotune_cu}"
                )
        logger.info(f"   • Header: {header_file}")
        logger.info(f"   • Utils Header: {utils_header_file}")

    # Read all source files
    naive_code, _, _ = get_cuda_code(
        str(naive_cu), str(header_file), str(utils_header_file)
    )
    coalesce_code, _, _ = get_cuda_code(
        str(coalesce_cu), str(header_file), str(utils_header_file)
    )
    shared_mem_code, _, _ = get_cuda_code(
        str(shared_mem_cu), str(header_file), str(utils_header_file)
    )
    blocktiling_1d_code, _, _ = get_cuda_code(
        str(blocktiling_1d_cu), str(header_file), str(utils_header_file)
    )
    blocktiling_2d_code, _, _ = get_cuda_code(
        str(blocktiling_2d_cu), str(header_file), str(utils_header_file)
    )
    vectorize_code, _, _ = get_cuda_code(
        str(vectorize_cu), str(header_file), str(utils_header_file)
    )
    warptiling_code, _, _ = get_cuda_code(
        str(warptiling_cu), str(header_file), str(utils_header_file)
    )
    warptiling_multidtype_code, _, _ = get_cuda_code(
        str(warptiling_multidtype_cu), str(header_file), str(utils_header_file)
    )
    tensorcore_naive_code, _, _ = get_cuda_code(
        str(tensorcore_naive_cu), str(header_file), str(utils_header_file)
    )
    tensorcore_code, _, _ = get_cuda_code(
        str(tensorcore_cu), str(header_file), str(utils_header_file)
    )
    tensorcore_double_buffered_code, _, _ = get_cuda_code(
        str(tensorcore_double_buffered_cu), str(header_file), str(utils_header_file)
    )
    tensorcore_async_code, _, _ = get_cuda_code(
        str(tensorcore_async_cu), str(header_file), str(utils_header_file)
    )
    cutlass_code, _, _ = get_cuda_code(
        str(cutlass_cu), str(header_file), str(utils_header_file)
    )

    # Conditionally load autotunable CUTLASS kernels
    if load_autotune_kernels:
        cutlass_autotune_code, _, _ = get_cuda_code(
            str(cutlass_autotune_cu), str(header_file), str(utils_header_file)
        )
    else:
        cutlass_autotune_code = ""

    # Conditionally load Hopper kernels based on GPU compute capability
    if has_hopper:
        cutlass_hopper_code, _, _ = get_cuda_code(
            str(cutlass_hopper_cu), str(header_file), str(utils_header_file)
        )
        if load_autotune_kernels:
            cutlass_hopper_autotune_code, header_code, utils_code = get_cuda_code(
                str(cutlass_hopper_autotune_cu),
                str(header_file),
                str(utils_header_file),
            )
        else:
            cutlass_hopper_autotune_code = ""
            _, header_code, utils_code = get_cuda_code(
                str(cutlass_hopper_cu), str(header_file), str(utils_header_file)
            )
    else:
        cutlass_hopper_code = ""
        cutlass_hopper_autotune_code = ""
        # Still need to read header and utils from one of the other files
        _, header_code, utils_code = get_cuda_code(
            str(cutlass_cu), str(header_file), str(utils_header_file)
        )

    # Combine CUDA sources
    # Add preprocessor directives to enable half-precision and WMMA for Tensor Cores
    # Must be at the top before any other includes
    # Build CUDA header with conditional Hopper includes
    cuda_header = """
// Undefine PyTorch's restrictive macros to enable Tensor Core operations
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif

// Now include CUDA headers for half precision and WMMA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

// Include cooperative groups and pipeline for async kernel
#include <cooperative_groups.h>
#include <cuda/pipeline>

// Include CUTLASS headers
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

"""

    # Add CUTLASS 3.x headers only if Hopper is available
    if has_hopper:
        cuda_header += """
// CUTLASS 3.x headers for Hopper
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

"""

    cuda_header += """
#include <iostream>
#include <type_traits>

// Namespace alias for cooperative groups (needed for async kernel)
namespace cg = cooperative_groups;

"""

    combined_cuda_code = (
        cuda_header
        + utils_code
        + "\n"
        + naive_code
        + "\n"
        + coalesce_code
        + "\n"
        + shared_mem_code
        + "\n"
        + blocktiling_1d_code
        + "\n"
        + blocktiling_2d_code
        + "\n"
        + vectorize_code
        + "\n"
        + warptiling_code
        + "\n"
        + warptiling_multidtype_code
        + "\n"
        + tensorcore_naive_code
        + "\n"
        + tensorcore_code
        + "\n"
        + tensorcore_double_buffered_code
        + "\n"
        + tensorcore_async_code
        + "\n"
        + cutlass_code
    )

    # Conditionally add autotunable CUTLASS kernels
    if load_autotune_kernels:
        combined_cuda_code += "\n" + cutlass_autotune_code

    # Conditionally add Hopper kernels if SM90+ is available
    if has_hopper:
        combined_cuda_code += "\n" + cutlass_hopper_code
        if load_autotune_kernels:
            combined_cuda_code += "\n" + cutlass_hopper_autotune_code

    # Create build directory
    build_dir = file_dir / "build" / "gemm_extension"
    build_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info(f"🔨 Build directory: {build_dir}")

    # Determine CUTLASS include paths from third-party directory
    cutlass_include_path = file_dir.parent / "third-party" / "cutlass" / "include"
    cutlass_utils_include_path = (
        file_dir.parent / "third-party" / "cutlass" / "tools" / "util" / "include"
    )

    if cutlass_include_path.exists():
        if verbose:
            logger.info(f"📦 Found CUTLASS headers at: {cutlass_include_path}")
    else:
        if verbose:
            logger.warning(
                f"⚠️  CUTLASS headers not found at {cutlass_include_path}. CUTLASS kernels may fail to compile."
            )

    if cutlass_utils_include_path.exists():
        if verbose:
            logger.info(
                f"📦 Found CUTLASS utils headers at: {cutlass_utils_include_path}"
            )
    else:
        if verbose:
            logger.warning(
                f"⚠️  CUTLASS utils headers not found at {cutlass_utils_include_path}. CUTLASS kernels may fail to compile."
            )

    # Prepare extra compiler flags
    extra_cflags = ["-O3", "-std=c++17"]
    # Add CUDA architecture flags
    # For Hopper (SM90a): enables TMA and warp specialization features
    # For non-Hopper: use sm_80 (Ampere) as baseline for tensor cores
    if has_hopper:
        extra_cuda_cflags = ["-O3", "-std=c++17", "--gpu-architecture=sm_90a"]
    else:
        extra_cuda_cflags = ["-O3", "-std=c++17"]
    extra_include_paths = [str(cutlass_include_path), str(cutlass_utils_include_path)]

    # Load the extension
    # Note: PyTorch adds -D__CUDA_NO_HALF_* macros by default, but we handle
    # them with #undef in the source code itself (see cuda_header above)
    # Build function list - conditionally include autotunable and Hopper functions
    functions_list = [
        "sgemm_naive",
        "sgemm_global_mem_coalesce",
        "sgemm_shared_mem",
        "sgemm_blocktiling_1d",
        "sgemm_blocktiling_2d",
        "sgemm_vectorize",
        "sgemm_warptiling_default",
        "sgemm_warptiling_fp16",
        "sgemm_warptiling_bf16",
        "sgemm_tensorcore_naive_fp16",
        "sgemm_tensorcore_naive_bf16",
        "sgemm_tensorcore_fp16",
        "sgemm_tensorcore_bf16",
        "sgemm_tensorcore_double_buffered_fp16",
        "sgemm_tensorcore_double_buffered_bf16",
        "sgemm_tensorcore_async_fp16",
        "sgemm_tensorcore_async_bf16",
        "sgemm_cutlass_fp16",
        "sgemm_cutlass_bf16",
        "sgemm_cutlass_fp32",
    ]

    # Add autotunable CUTLASS functions if requested
    if load_autotune_kernels:
        functions_list.extend(
            [
                "sgemm_cutlass_autotune_fp16",
                "sgemm_cutlass_autotune_bf16",
                "get_num_cutlass_configs",
            ]
        )

    # Add Hopper functions only if SM90+ is available (BF16 only, FP16 not supported)
    if has_hopper:
        functions_list.extend(
            [
                "sgemm_cutlass_hopper_bf16",
                "sgemm_cutlass_hopper_bf16_tma_warp_specialized_auto",
                "sgemm_cutlass_hopper_bf16_tma_warp_specialized_constant",
                "sgemm_cutlass_hopper_bf16_tma_warp_specialized_persistent_constant",
                "sgemm_cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant",
                "sgemm_cutlass_hopper_bf16_tma_warp_specialized_streamk_constant",
            ]
        )
        # Add autotunable Hopper functions if requested
        if load_autotune_kernels:
            functions_list.extend(
                [
                    "sgemm_cutlass_hopper_autotune_bf16",
                    "get_num_cutlass_hopper_configs",
                ]
            )

    extension = load_inline(
        name="gemm_cuda_extension",
        cpp_sources=header_code,
        cuda_sources=combined_cuda_code,
        functions=functions_list,
        with_cuda=True,
        verbose=verbose,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        build_directory=str(build_dir),
    )

    if verbose:
        logger.success("✅ CUDA extension loaded successfully!")

    return extension
