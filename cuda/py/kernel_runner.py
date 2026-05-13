"""Run CUDA GEMM kernels for NCU profiling.

This script allows you to dry run any registered kernel N times for profiling with NVIDIA Nsight Compute (ncu).

Usage:
    # Run naive kernel 100 times with default matrix size (1024)
    python kernel_runner.py -k naive -n 100

    # Run with custom matrix size
    python kernel_runner.py -k shared_mem -n 50 -s 2048

    # Profile with ncu
    ncu --set full python kernel_runner.py -k naive -n 100
    ncu --metrics all python kernel_runner.py -k global_mem_coalesce -n 100 -s 512

Available kernels:
    - pytorch: PyTorch baseline implementation (torch.matmul)
    - naive: Naive CUDA GEMM kernel
    - global_mem_coalesce: CUDA GEMM with global memory coalescing
    - shared_mem: CUDA GEMM with shared memory tiling
    - blocktiling_1d: CUDA GEMM with 1D block tiling
    - blocktiling_2d: CUDA GEMM with 2D block tiling
    - vectorize: CUDA GEMM with vectorized memory access
    - warptiling: CUDA GEMM with warp-level tiling (most optimized FP32)
    - tensorcore_naive_fp16: Naive Tensor Core with FP16 inputs
    - tensorcore_naive_bf16: Naive Tensor Core with BF16 inputs
    - tensorcore_fp16: CUDA Tensor Core (warptiled) with FP16 inputs
    - tensorcore_bf16: CUDA Tensor Core (warptiled) with BF16 inputs
    - tensorcore_db_fp16: CUDA Tensor Core with double buffering (FP16)
    - tensorcore_db_bf16: CUDA Tensor Core with double buffering (BF16)
    - tensorcore_async_fp16: CUDA Tensor Core with async pipeline (FP16)
    - tensorcore_async_bf16: CUDA Tensor Core with async pipeline (BF16)
    - cutlass_fp16: CUTLASS library GEMM with FP16 inputs
    - cutlass_bf16: CUTLASS library GEMM with BF16 inputs
    - cutlass_fp32: CUTLASS library GEMM with FP32 inputs (SIMT)
    - cutlass_hopper_bf16: CUTLASS Hopper GEMM with BF16 (SM90+, default variant)
    - cutlass_hopper_bf16_tma_warp_specialized_auto: CUTLASS Hopper TMA Warp Specialized Auto
    - cutlass_hopper_bf16_tma_warp_specialized_constant: CUTLASS Hopper TMA Warp Specialized Constant
    - cutlass_hopper_bf16_tma_warp_specialized_persistent_constant: CUTLASS Hopper TMA Persistent Constant
    - cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant: CUTLASS Hopper TMA Pingpong Constant
    - cutlass_hopper_bf16_tma_warp_specialized_streamk_constant: CUTLASS Hopper TMA Stream-K Constant
"""

import os
from typing import Callable

import click

# Set CUDA paths to match CMakeLists.txt configuration
# Must be set BEFORE importing torch
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import torch
from loguru import logger

# Import the shared CUDA extension loader
from cuda_extension_loader import create_cuda_extension


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Load CUDA kernels (with less verbose output for kernel runner)
logger.info("🚀 Loading CUDA kernels...")
cuda_kernels = create_cuda_extension(verbose=True, load_hopper_kernels=True)
logger.success("✅ CUDA kernels loaded successfully!")


def run_pytorch_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run PyTorch baseline implementation."""
    return torch.matmul(a, b, out=c)


def run_naive_kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Run naive CUDA GEMM kernel."""
    cuda_kernels.sgemm_naive(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_coalesced_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run coalesced global memory CUDA GEMM kernel."""
    cuda_kernels.sgemm_global_mem_coalesce(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_shared_mem_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run shared memory CUDA GEMM kernel."""
    cuda_kernels.sgemm_shared_mem(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_blocktiling_1d_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run 1D block tiling CUDA GEMM kernel."""
    cuda_kernels.sgemm_blocktiling_1d(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_blocktiling_2d_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run 2D block tiling CUDA GEMM kernel."""
    cuda_kernels.sgemm_blocktiling_2d(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_vectorize_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run vectorized CUDA GEMM kernel."""
    cuda_kernels.sgemm_vectorize(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_warptiling_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run warptiling CUDA GEMM kernel (dtype-aware)."""
    # Dispatch to appropriate warptiling kernel based on input dtype
    if a.dtype == torch.float32:
        cuda_kernels.sgemm_warptiling_default(a, b, c, 1.0, 0.0)  # type: ignore
    elif a.dtype == torch.float16:
        cuda_kernels.sgemm_warptiling_fp16(a, b, c, 1.0, 0.0)  # type: ignore
    elif a.dtype == torch.bfloat16:
        cuda_kernels.sgemm_warptiling_bf16(a, b, c, 1.0, 0.0)  # type: ignore
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")

    return c


def run_tensorcore_naive_fp16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run naive Tensor Core CUDA GEMM kernel with FP16 inputs."""
    cuda_kernels.sgemm_tensorcore_naive_fp16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_naive_bf16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run naive Tensor Core CUDA GEMM kernel with BF16 inputs."""
    cuda_kernels.sgemm_tensorcore_naive_bf16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_fp16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run optimized Tensor Core CUDA GEMM kernel with FP16 inputs."""
    cuda_kernels.sgemm_tensorcore_fp16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_bf16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run optimized Tensor Core CUDA GEMM kernel with BF16 inputs."""
    cuda_kernels.sgemm_tensorcore_bf16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_db_fp16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run Tensor Core CUDA GEMM kernel with FP16 inputs and double buffering."""
    cuda_kernels.sgemm_tensorcore_double_buffered_fp16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_db_bf16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run Tensor Core CUDA GEMM kernel with BF16 inputs and double buffering."""
    cuda_kernels.sgemm_tensorcore_double_buffered_bf16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_async_fp16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run Tensor Core CUDA GEMM kernel with FP16 inputs and async pipeline."""
    cuda_kernels.sgemm_tensorcore_async_fp16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_async_bf16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run Tensor Core CUDA GEMM kernel with BF16 inputs and async pipeline."""
    cuda_kernels.sgemm_tensorcore_async_bf16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_cutlass_fp16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS GEMM kernel with FP16 inputs."""
    cuda_kernels.sgemm_cutlass_fp16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_cutlass_bf16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS GEMM kernel with BF16 inputs."""
    cuda_kernels.sgemm_cutlass_bf16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_cutlass_fp32_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS GEMM kernel with FP32 inputs."""
    cuda_kernels.sgemm_cutlass_fp32(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_cutlass_hopper_bf16_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS Hopper GEMM kernel with BF16 inputs (SM90+) - Default variant."""
    cuda_kernels.sgemm_cutlass_hopper_bf16(a, b, c)  # type: ignore
    return c


def run_cutlass_hopper_bf16_tma_warp_specialized_auto_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS Hopper TMA Warp Specialized Auto kernel with BF16 inputs (SM90+)."""
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_auto(a, b, c)  # type: ignore
    return c


def run_cutlass_hopper_bf16_tma_warp_specialized_constant_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS Hopper TMA Warp Specialized Constant kernel with BF16 inputs (SM90+)."""
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_constant(a, b, c)  # type: ignore
    return c


def run_cutlass_hopper_bf16_tma_warp_specialized_persistent_constant_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS Hopper TMA Warp Specialized Persistent Constant kernel with BF16 inputs (SM90+)."""
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_persistent_constant(a, b, c)  # type: ignore
    return c


def run_cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS Hopper TMA Warp Specialized Pingpong Constant kernel with BF16 inputs (SM90+)."""
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant(a, b, c)  # type: ignore
    return c


def run_cutlass_hopper_bf16_tma_warp_specialized_streamk_constant_kernel(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Run CUTLASS Hopper TMA Warp Specialized Stream-K Constant kernel with BF16 inputs (SM90+)."""
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_streamk_constant(a, b, c)  # type: ignore
    return c


def run_kernel_n_times(
    kernel_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n: int,
) -> None:
    """Run a kernel N times for profiling."""
    logger.info(f"⚡ Running kernel {n} times...")

    for _ in range(n):
        _ = kernel_fn(a, b, c)

    # Synchronize to ensure all kernels complete
    torch.cuda.synchronize()
    logger.success(f"✅ Completed {n} iterations")


@click.command()
@click.option(
    "-k",
    "--kernel",
    type=click.Choice(
        [
            "pytorch",
            "naive",
            "global_mem_coalesce",
            "shared_mem",
            "blocktiling_1d",
            "blocktiling_2d",
            "vectorize",
            "warptiling",
            "tensorcore_naive_fp16",
            "tensorcore_naive_bf16",
            "tensorcore_fp16",
            "tensorcore_bf16",
            "tensorcore_db_fp16",
            "tensorcore_db_bf16",
            "tensorcore_async_fp16",
            "tensorcore_async_bf16",
            "cutlass_fp16",
            "cutlass_bf16",
            "cutlass_fp32",
            "cutlass_hopper_bf16",
            "cutlass_hopper_bf16_tma_warp_specialized_auto",
            "cutlass_hopper_bf16_tma_warp_specialized_constant",
            "cutlass_hopper_bf16_tma_warp_specialized_persistent_constant",
            "cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant",
            "cutlass_hopper_bf16_tma_warp_specialized_streamk_constant",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="Kernel to run",
)
@click.option(
    "-n",
    "--iterations",
    type=int,
    default=100,
    help="Number of times to run the kernel (default: 100)",
)
@click.option(
    "-s",
    "--size",
    type=int,
    default=4096,
    help="Matrix size (M=N=K, default: 4096)",
)
@click.option(
    "-d",
    "--dtype",
    type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
    default="float32",
    help="Data type for input matrices (default: float32). Tensor Core kernels require float16 or bfloat16.",
)
def main(kernel: str, iterations: int, size: int, dtype: str):
    """Run CUDA GEMM kernels for NCU profiling.

    Examples:
        # Run naive kernel 100 times
        python kernel_runner.py -k naive -n 100

        # Run with FP16 and Tensor Cores
        python kernel_runner.py -k tensorcore_fp16 -n 100 -d float16

        # Run with custom matrix size
        python kernel_runner.py -k shared_mem -n 50 -s 2048

        # Profile with ncu
        ncu --set full python kernel_runner.py -k naive -n 100
        ncu --metrics all python kernel_runner.py -k tensorcore_fp16 -n 100 -d float16
    """
    # Map kernel names to functions
    kernel_map = {
        "pytorch": run_pytorch_kernel,
        "naive": run_naive_kernel,
        "global_mem_coalesce": run_coalesced_kernel,
        "shared_mem": run_shared_mem_kernel,
        "blocktiling_1d": run_blocktiling_1d_kernel,
        "blocktiling_2d": run_blocktiling_2d_kernel,
        "vectorize": run_vectorize_kernel,
        "warptiling": run_warptiling_kernel,
        "tensorcore_naive_fp16": run_tensorcore_naive_fp16_kernel,
        "tensorcore_naive_bf16": run_tensorcore_naive_bf16_kernel,
        "tensorcore_fp16": run_tensorcore_fp16_kernel,
        "tensorcore_bf16": run_tensorcore_bf16_kernel,
        "tensorcore_db_fp16": run_tensorcore_db_fp16_kernel,
        "tensorcore_db_bf16": run_tensorcore_db_bf16_kernel,
        "tensorcore_async_fp16": run_tensorcore_async_fp16_kernel,
        "tensorcore_async_bf16": run_tensorcore_async_bf16_kernel,
        "cutlass_fp16": run_cutlass_fp16_kernel,
        "cutlass_bf16": run_cutlass_bf16_kernel,
        "cutlass_fp32": run_cutlass_fp32_kernel,
        "cutlass_hopper_bf16": run_cutlass_hopper_bf16_kernel,
        "cutlass_hopper_bf16_tma_warp_specialized_auto": run_cutlass_hopper_bf16_tma_warp_specialized_auto_kernel,
        "cutlass_hopper_bf16_tma_warp_specialized_constant": run_cutlass_hopper_bf16_tma_warp_specialized_constant_kernel,
        "cutlass_hopper_bf16_tma_warp_specialized_persistent_constant": run_cutlass_hopper_bf16_tma_warp_specialized_persistent_constant_kernel,
        "cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant": run_cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant_kernel,
        "cutlass_hopper_bf16_tma_warp_specialized_streamk_constant": run_cutlass_hopper_bf16_tma_warp_specialized_streamk_constant_kernel,
    }

    # Auto-detect required dtype for Tensor Core and CUTLASS kernels if not specified
    if (
        kernel
        in [
            "tensorcore_naive_fp16",
            "tensorcore_fp16",
            "tensorcore_db_fp16",
            "tensorcore_async_fp16",
            "cutlass_fp16",
        ]
        and dtype == "float32"
    ):
        logger.warning(f"⚠️  {kernel} requires float16 dtype, auto-switching to float16")
        dtype = "float16"
    elif (
        kernel
        in [
            "tensorcore_naive_bf16",
            "tensorcore_bf16",
            "tensorcore_db_bf16",
            "tensorcore_async_bf16",
            "cutlass_bf16",
            "cutlass_hopper_bf16",
            "cutlass_hopper_bf16_tma_warp_specialized_auto",
            "cutlass_hopper_bf16_tma_warp_specialized_constant",
            "cutlass_hopper_bf16_tma_warp_specialized_persistent_constant",
            "cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant",
            "cutlass_hopper_bf16_tma_warp_specialized_streamk_constant",
        ]
        and dtype == "float32"
    ):
        logger.warning(
            f"⚠️  {kernel} requires bfloat16 dtype, auto-switching to bfloat16"
        )
        dtype = "bfloat16"

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    kernel_fn = kernel_map[kernel]
    M = N = K = size

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Kernel: {kernel}")
    logger.info(f"📐 Matrix size: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    logger.info(f"🔢 Data type: {dtype}")
    logger.info(f"🔄 Iterations: {iterations}")
    logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"{'='*60}\n")

    # Validate dtype compatibility with kernel
    fp32_only_kernels = [
        "naive",
        "global_mem_coalesce",
        "shared_mem",
        "blocktiling_1d",
        "blocktiling_2d",
        "vectorize",
    ]

    if kernel in fp32_only_kernels and dtype != "float32":
        logger.error(
            f"❌ Kernel '{kernel}' only supports float32, but {dtype} was specified"
        )
        logger.error(f"   Please use --dtype=float32 or choose a different kernel")
        return

    # Create input tensors with specified dtype
    logger.info("💾 Allocating input tensors...")
    a = torch.randn((M, K), device="cuda", dtype=torch_dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch_dtype)

    # Determine output dtype based on kernel type
    # - PyTorch and warptiling: output matches input dtype
    # - Tensor Core and CUTLASS kernels: output FP32 (they accumulate in FP32)
    # - CUTLASS Hopper BF16: output BF16 (matches input dtype)
    # - FP32-only kernels: output FP32
    if kernel in ["pytorch", "warptiling"]:
        output_dtype = torch_dtype
    elif kernel in [
        "cutlass_hopper_bf16",
        "cutlass_hopper_bf16_tma_warp_specialized_auto",
        "cutlass_hopper_bf16_tma_warp_specialized_constant",
        "cutlass_hopper_bf16_tma_warp_specialized_persistent_constant",
        "cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant",
        "cutlass_hopper_bf16_tma_warp_specialized_streamk_constant",
    ]:
        # Hopper kernels expect BF16 output to match input
        output_dtype = torch_dtype
    elif kernel in [
        "tensorcore_naive_fp16",
        "tensorcore_naive_bf16",
        "tensorcore_fp16",
        "tensorcore_bf16",
        "tensorcore_db_fp16",
        "tensorcore_db_bf16",
        "tensorcore_async_fp16",
        "tensorcore_async_bf16",
        "cutlass_fp16",
        "cutlass_bf16",
        "cutlass_fp32",
    ]:
        output_dtype = torch.float32
    else:
        # FP32-only kernels
        output_dtype = torch.float32

    c = torch.empty((M, N), device="cuda", dtype=output_dtype)
    logger.success("✅ Input and output tensors allocated")
    logger.info(f"   Input dtype: {torch_dtype}, Output dtype: {output_dtype}")

    # Run the kernel N times
    logger.info("")
    run_kernel_n_times(kernel_fn, a, b, c, iterations)

    logger.info(f"\n{'='*60}")
    logger.success("🎉 Done!")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
