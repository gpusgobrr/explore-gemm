"""Benchmark CUDA GEMM kernels against PyTorch with visualization.

Usage:
    # Run all FP32 kernels (default)
    python benchmark.py

    # Run all FP16-compatible kernels (PyTorch, warptiling, tensorcore)
    python benchmark.py -d float16

    # Run all BF16-compatible kernels (PyTorch, warptiling, tensorcore)
    python benchmark.py -d bfloat16

    # Run specific kernels with FP16
    python benchmark.py -d float16 -k pytorch -k tensorcore_fp16

    # Run specific kernels with FP32
    python benchmark.py -k pytorch -k naive
    python benchmark.py --kernels naive --kernels global_mem_coalesce --kernels shared_mem

Available kernels:
    All dtypes (FP32/FP16/BF16):
    - pytorch: PyTorch baseline implementation
    - warptiling: CUDA GEMM with warp-level tiling (supports all dtypes)

    FP32 only:
    - naive: Naive CUDA GEMM kernel
    - global_mem_coalesce: CUDA GEMM with global memory coalescing
    - shared_mem: CUDA GEMM with shared memory tiling
    - blocktiling_1d: CUDA GEMM with 1D block tiling
    - blocktiling_2d: CUDA GEMM with 2D block tiling
    - vectorize: CUDA GEMM with vectorized memory access
    - cutlass_fp32: CUTLASS library GEMM with FP32 inputs (SIMT operations)

    FP16/BF16 only:
    - tensorcore_fp16: CUDA Tensor Core with FP16 inputs (requires -d float16)
    - tensorcore_bf16: CUDA Tensor Core with BF16 inputs (requires -d bfloat16)
    - tensorcore_db_fp16: CUDA Tensor Core with double buffering (FP16)
    - tensorcore_db_bf16: CUDA Tensor Core with double buffering (BF16)
    - tensorcore_async_fp16: CUDA Tensor Core with async pipeline (FP16)
    - tensorcore_async_bf16: CUDA Tensor Core with async pipeline (BF16)
    - cutlass_fp16: CUTLASS library GEMM with FP16 inputs (requires -d float16)
    - cutlass_bf16: CUTLASS library GEMM with BF16 inputs (requires -d bfloat16)
    - cutlass_hopper_bf16: CUTLASS Hopper GEMM with BF16 (SM90+, requires -d bfloat16, default variant)
    - cutlass_hopper_bf16_tma_warp_specialized_auto: CUTLASS Hopper TMA Warp Specialized Auto
    - cutlass_hopper_bf16_tma_warp_specialized_constant: CUTLASS Hopper TMA Warp Specialized Constant
    - cutlass_hopper_bf16_tma_warp_specialized_persistent_constant: CUTLASS Hopper TMA Persistent Constant
    - cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant: CUTLASS Hopper TMA Pingpong Constant
    - cutlass_hopper_bf16_tma_warp_specialized_streamk_constant: CUTLASS Hopper TMA Stream-K Constant

Note: When using -d float16 or -d bfloat16 without specifying kernels, FP32-only
kernels are automatically filtered out. If you explicitly request FP32-only kernels
with FP16/BF16, they will be skipped with a warning.
"""

import os
from pathlib import Path
from typing import List
from datetime import datetime
import webbrowser

# Set CUDA paths to match CMakeLists.txt configuration
# Must be set BEFORE importing torch
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import torch
import numpy as np
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import click

# Import the shared CUDA extension loader
from cuda_extension_loader import create_cuda_extension


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Cache-flushing tensor size (16MB to flush typical L2 caches)
_CACHE_FLUSH_SIZE = 4 * 1024 * 1024  # 16MB in float32 elements

# Global cache flush buffer (lazily initialized)
_cache_flush_buffer = None


def get_cache_flush_buffer():
    """Get or create the cache flush buffer."""
    global _cache_flush_buffer
    if _cache_flush_buffer is None:
        _cache_flush_buffer = torch.empty(
            _CACHE_FLUSH_SIZE, dtype=torch.int8, device="cuda"
        )
    return _cache_flush_buffer


def flush_l2_cache():
    """Flush the L2 cache by performing a dummy operation on a large buffer.

    This helps eliminate cache state artifacts between benchmark iterations.
    """
    cache_buffer = get_cache_flush_buffer()
    # Perform a simple operation to flush cache
    cache_buffer.zero_()
    torch.cuda.synchronize()


# Load CUDA kernels
logger.info("🚀 Loading CUDA kernels...")
cuda_kernels = create_cuda_extension(verbose=True, load_hopper_kernels=True)


def cuda_naive_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for naive CUDA GEMM kernel."""
    # Create output tensor on CUDA device
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_naive(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def cuda_coalesced_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for coalesced global memory CUDA GEMM kernel."""
    # Create output tensor on CUDA device
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_global_mem_coalesce(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def cuda_shared_mem_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for shared memory CUDA GEMM kernel."""
    # Create output tensor on CUDA device
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_shared_mem(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def cuda_blocktiling_1d_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for 1D block tiling CUDA GEMM kernel."""
    # Create output tensor on CUDA device
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_blocktiling_1d(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def cuda_blocktiling_2d_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for 2D block tiling CUDA GEMM kernel."""
    # Create output tensor on CUDA device
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_blocktiling_2d(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def cuda_vectorize_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for vectorized CUDA GEMM kernel."""
    # Create output tensor on CUDA device
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_vectorize(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def cuda_warptiling_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for warptiling CUDA GEMM kernel (dtype-aware)."""
    # Create output tensor on CUDA device with same dtype as input (like PyTorch)
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=a.dtype)

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


def cuda_tensorcore_naive_fp16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for naive Tensor Core CUDA GEMM kernel with FP16 inputs.

    This is a baseline implementation without optimizations.
    """
    # Tensor Cores output FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_naive_fp16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_tensorcore_naive_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for naive Tensor Core CUDA GEMM kernel with BF16 inputs.

    This is a baseline implementation without optimizations.
    """
    # Tensor Cores output FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_naive_bf16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_tensorcore_fp16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for Tensor Core CUDA GEMM kernel with FP16 inputs.

    Note: Tensor Cores accumulate in FP32 for numerical stability, so output is FP32.
    This differs from PyTorch which returns FP16. To match PyTorch behavior exactly,
    we convert the output back to FP16.
    """
    # Tensor Cores output FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_fp16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_tensorcore_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for Tensor Core CUDA GEMM kernel with BF16 inputs.

    Note: Tensor Cores accumulate in FP32 for numerical stability, so output is FP32.
    This differs from PyTorch which returns BF16. To match PyTorch behavior exactly,
    we convert the output back to BF16.
    """
    # Tensor Cores output FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_bf16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_tensorcore_db_fp16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for Tensor Core CUDA GEMM kernel with FP16 inputs and double buffering.

    Double buffering overlaps memory loads with computation for better performance.
    """
    # Tensor Cores output FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_double_buffered_fp16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_tensorcore_db_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for Tensor Core CUDA GEMM kernel with BF16 inputs and double buffering.

    Double buffering overlaps memory loads with computation for better performance.
    """
    # Tensor Cores output FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_double_buffered_bf16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_tensorcore_async_fp16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for Tensor Core CUDA GEMM kernel with FP16 inputs and async pipeline.

    Async pipeline uses cp.async for maximum memory/compute overlap.
    Requires SM 8.0+ (Ampere and newer).
    """
    # Tensor Cores output FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_async_fp16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_tensorcore_async_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for Tensor Core CUDA GEMM kernel with BF16 inputs and async pipeline.

    Async pipeline uses cp.async for maximum memory/compute overlap.
    Requires SM 8.0+ (Ampere and newer).
    """
    # Tensor Cores output FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_async_bf16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_cutlass_fp16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for CUTLASS GEMM kernel with FP16 inputs.

    Uses NVIDIA CUTLASS library for highly optimized Tensor Core operations.
    """
    # CUTLASS outputs FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_cutlass_fp16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_cutlass_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for CUTLASS GEMM kernel with BF16 inputs.

    Uses NVIDIA CUTLASS library for highly optimized Tensor Core operations.
    """
    # CUTLASS outputs FP32 for precision
    c_fp32 = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_cutlass_bf16(a, b, c_fp32, 1.0, 0.0)  # type: ignore
    # Convert to input dtype to match PyTorch behavior
    return c_fp32.to(a.dtype)


def cuda_cutlass_fp32_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for CUTLASS GEMM kernel with FP32 inputs.

    Uses NVIDIA CUTLASS library with SIMT operations for FP32.
    """
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_cutlass_fp32(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def cuda_cutlass_hopper_bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for CUTLASS Hopper GEMM kernel with BF16 inputs (default variant).

    Uses NVIDIA CUTLASS 3.x Collective Builder API for Hopper (SM90+) with warp specialization.
    Default: Pingpong with constant stage count.
    """
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.bfloat16)
    cuda_kernels.sgemm_cutlass_hopper_bf16(a, b, c)  # type: ignore
    return c


def cuda_cutlass_hopper_bf16_tma_warp_specialized_auto_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUTLASS Hopper: TMA Warp Specialized with Auto stage count."""
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.bfloat16)
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_auto(a, b, c)  # type: ignore
    return c


def cuda_cutlass_hopper_bf16_tma_warp_specialized_constant_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUTLASS Hopper: TMA Warp Specialized with Constant stage count."""
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.bfloat16)
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_constant(a, b, c)  # type: ignore
    return c


def cuda_cutlass_hopper_bf16_tma_warp_specialized_persistent_constant_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUTLASS Hopper: TMA Warp Specialized Persistent with Constant stage count."""
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.bfloat16)
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_persistent_constant(a, b, c)  # type: ignore
    return c


def cuda_cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUTLASS Hopper: TMA Warp Specialized Pingpong with Constant stage count."""
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.bfloat16)
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant(a, b, c)  # type: ignore
    return c


def cuda_cutlass_hopper_bf16_tma_warp_specialized_streamk_auto_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUTLASS Hopper: TMA Warp Specialized Stream-K with Auto stage count."""
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.bfloat16)
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_streamk_auto(a, b, c)  # type: ignore
    return c


def cuda_cutlass_hopper_bf16_tma_warp_specialized_streamk_constant_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUTLASS Hopper: TMA Warp Specialized Stream-K with Constant stage count."""
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.bfloat16)
    cuda_kernels.sgemm_cutlass_hopper_bf16_tma_warp_specialized_streamk_constant(a, b, c)  # type: ignore
    return c


def torch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation.

    Preallocates output tensor to match the behavior of custom CUTLASS kernels
    for fair performance comparison.
    """
    # Preallocate output tensor (same dtype as input)
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=a.dtype)
    torch.matmul(a, b, out=c)
    return c


def benchmark_kernel(kernel_fn, a, b, warmup=10, iterations=100, flush_cache=True):
    """Benchmark a GEMM kernel function.

    Args:
        kernel_fn: Kernel function to benchmark
        a: Input matrix A
        b: Input matrix B
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        flush_cache: Whether to flush L2 cache between iterations

    Returns:
        Tuple of (median_time_ms, min_time_ms, max_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        if flush_cache:
            flush_l2_cache()
        _ = kernel_fn(a, b)

    # Benchmark
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        # Flush L2 cache before each iteration to eliminate cache state artifacts
        if flush_cache:
            flush_l2_cache()

        start_events[i].record()
        _ = kernel_fn(a, b)
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate times
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Trim outliers (top and bottom 10%)
    times_ms_sorted = sorted(times_ms)
    trim_count = max(1, iterations // 10)
    times_ms_trimmed = times_ms_sorted[trim_count:-trim_count]

    # Use median instead of average for more robust comparison
    median_time_ms = float(np.median(times_ms_trimmed))
    min_time_ms = min(times_ms_trimmed)
    max_time_ms = max(times_ms_trimmed)

    return median_time_ms, min_time_ms, max_time_ms


def calculate_metrics(M, N, K, avg_time_ms, element_size: int = 4):
    """Calculate TFLOPS and bandwidth.

    Args:
        M, N, K: Matrix dimensions
        avg_time_ms: Average execution time in milliseconds
        element_size: Bytes per element (4 for FP32, 2 for FP16/BF16)
    """
    # FLOPs: 2MNK for matrix multiplication (2 ops per MAC)
    flops = 2 * M * N * K
    tflops = (flops / (avg_time_ms * 1e-3)) * 1e-12

    # Memory bandwidth: read A (MxK), read B (KxN), write C (MxN)
    # All matrices use the same dtype now (matching PyTorch behavior)
    bytes_total = (M * K + K * N + M * N) * element_size
    bandwidth_gbps = (bytes_total * 1e-9) / (avg_time_ms * 1e-3)

    return tflops, bandwidth_gbps


def create_visualization(
    results_df: pd.DataFrame, output_dir: Path, dtype: str = "float32"
):
    """Create interactive Plotly visualizations."""
    logger.info("📊 Creating visualizations...")

    # Format dtype for display
    dtype_display = (
        dtype.upper()
        if dtype == "float32"
        else dtype.replace("float", "FP").replace("bfloat", "BF").upper()
    )

    # Determine which TFLOP metric to use
    tflop_metric = "TFLOP16s" if dtype in ["float16", "bfloat16"] else "TFLOPS"

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"🚀 {tflop_metric} Performance vs Matrix Size ({dtype_display})",
            f"💾 Memory Bandwidth vs Matrix Size ({dtype_display})",
            f"⏱️ Execution Time vs Matrix Size ({dtype_display})",
            f"📈 Speedup vs PyTorch (Baseline) ({dtype_display})",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    kernels = results_df["kernel"].unique()
    colors = {
        "PyTorch": "#636EFA",
        "CUDA Naive": "#EF553B",
        "CUDA Coalesced": "#00CC96",
        "CUDA Shared Mem": "#AB63FA",
        "CUDA 1D Block Tiling": "#FFA15A",
        "CUDA 2D Block Tiling": "#19D3F3",
        "CUDA Vectorize": "#FF6692",
        "CUDA Warptiling": "#FEC200",
        "CUDA Tensor Core Naive (FP16)": "#00CED1",  # Dark Turquoise
        "CUDA Tensor Core Naive (BF16)": "#00CED1",  # Dark Turquoise (same as FP16)
        "CUDA Tensor Core Warptiled (FP16)": "#FF1493",  # Deep Pink
        "CUDA Tensor Core Warptiled (BF16)": "#FF1493",  # Deep Pink (same as FP16)
        "CUDA Tensor Core Double Buffered (FP16)": "#32CD32",  # Lime Green
        "CUDA Tensor Core Double Buffered (BF16)": "#32CD32",  # Lime Green (same as FP16)
        "CUTLASS (FP16)": "#9370DB",  # Medium Purple
        "CUTLASS (BF16)": "#9370DB",  # Medium Purple (same as FP16)
        "CUTLASS (FP32)": "#DC143C",  # Crimson
    }

    # Plot 1: TFLOPS
    for kernel in kernels:
        kernel_data = results_df[results_df["kernel"] == kernel]
        fig.add_trace(
            go.Scatter(
                x=kernel_data["size"],
                y=kernel_data["tflops"],
                name=kernel,
                mode="lines+markers",
                line=dict(color=colors.get(kernel, "#000000"), width=2),
                marker=dict(size=8),
                legendgroup=kernel,
            ),
            row=1,
            col=1,
        )

    # Plot 2: Bandwidth
    for kernel in kernels:
        kernel_data = results_df[results_df["kernel"] == kernel]
        fig.add_trace(
            go.Scatter(
                x=kernel_data["size"],
                y=kernel_data["bandwidth_gbps"],
                name=kernel,
                mode="lines+markers",
                line=dict(color=colors.get(kernel, "#000000"), width=2),
                marker=dict(size=8),
                legendgroup=kernel,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Plot 3: Execution Time
    for kernel in kernels:
        kernel_data = results_df[results_df["kernel"] == kernel]
        fig.add_trace(
            go.Scatter(
                x=kernel_data["size"],
                y=kernel_data["avg_time_ms"],
                name=kernel,
                mode="lines+markers",
                line=dict(color=colors.get(kernel, "#000000"), width=2),
                marker=dict(size=8),
                legendgroup=kernel,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Plot 4: Speedup
    for kernel in kernels:
        if kernel == "PyTorch":
            continue
        kernel_data = results_df[results_df["kernel"] == kernel]
        fig.add_trace(
            go.Scatter(
                x=kernel_data["size"],
                y=kernel_data["speedup"],
                name=kernel,
                mode="lines+markers",
                line=dict(color=colors.get(kernel, "#000000"), width=2),
                marker=dict(size=8),
                legendgroup=kernel,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Add baseline line for speedup
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="PyTorch Baseline",
        row=2,  # type: ignore
        col=2,  # type: ignore
    )

    # Update axes
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", row=1, col=1, type="log")
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", row=1, col=2, type="log")
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", row=2, col=1, type="log")
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", row=2, col=2, type="log")

    fig.update_yaxes(title_text=tflop_metric, row=1, col=1)
    fig.update_yaxes(title_text="Bandwidth (GB/s)", row=1, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=1, type="log")
    fig.update_yaxes(title_text="Speedup (×)", row=2, col=2)

    # Update layout
    fig.update_layout(
        height=900,
        title_text=f"<b>CUDA GEMM Kernel Benchmarks ({dtype_display})</b>",
        title_x=0.5,
        title_font=dict(size=20),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )

    # Save interactive HTML
    html_file = output_dir / "benchmark_results.html"
    fig.write_html(str(html_file))
    logger.success(f"✅ Saved interactive visualization to {html_file}")

    return fig


def create_html_report(
    results_df: pd.DataFrame, output_dir: Path, dtype: str = "float32"
):
    """Create HTML report with results and visualizations."""
    logger.info("📝 Creating HTML report...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    # Format dtype for display
    dtype_display = (
        dtype.upper()
        if dtype == "float32"
        else dtype.replace("float", "FP").replace("bfloat", "BF").upper()
    )

    # Determine which TFLOP metric to use
    tflop_metric = "TFLOP16s" if dtype in ["float16", "bfloat16"] else "TFLOPS"

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CUDA GEMM Benchmark Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 3px solid #e9ecef;
        }}

        .info-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .info-card h3 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .info-card p {{
            color: #333;
            font-size: 1.3em;
            font-weight: bold;
        }}

        .chart-container {{
            padding: 30px;
        }}

        .chart-container iframe {{
            width: 100%;
            height: 950px;
            border: none;
            border-radius: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px;
            max-width: calc(100% - 60px);
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}

        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .kernel-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-pytorch {{
            background: #e3f2fd;
            color: #1976d2;
        }}

        .badge-pytorch-jit {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}

        .badge-naive {{
            background: #ffebee;
            color: #c62828;
        }}

        .badge-coalesced {{
            background: #e8f5e9;
            color: #2e7d32;
        }}

        .badge-shared {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}

        .badge-blocktiling {{
            background: #fff3e0;
            color: #e65100;
        }}

        .badge-blocktiling2d {{
            background: #e0f7fa;
            color: #006064;
        }}

        .badge-vectorize {{
            background: #fce4ec;
            color: #c2185b;
        }}

        .badge-warptiling {{
            background: #fff9c4;
            color: #f57f17;
        }}

        .badge-tensorcore {{
            background: #e0f2f1;
            color: #00695c;
        }}

        .badge-cutlass {{
            background: #ffe0b2;
            color: #e65100;
        }}

        .speedup {{
            font-weight: bold;
        }}

        .speedup.faster {{
            color: #2e7d32;
        }}

        .speedup.slower {{
            color: #c62828;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 CUDA GEMM Kernel Benchmarks ({dtype_display})</h1>
            <p class="subtitle">Performance Comparison: PyTorch vs Custom CUDA Kernels</p>
        </header>

        <div class="info-grid">
            <div class="info-card">
                <h3>🖥️ GPU</h3>
                <p>{gpu_name}</p>
            </div>
            <div class="info-card">
                <h3>📊 Data Type</h3>
                <p>{dtype_display}</p>
            </div>
            <div class="info-card">
                <h3>⏰ Timestamp</h3>
                <p>{timestamp}</p>
            </div>
            <div class="info-card">
                <h3>🔢 Test Sizes</h3>
                <p>{len(results_df['size'].unique())} configurations</p>
            </div>
            <div class="info-card">
                <h3>⚡ Kernels Tested</h3>
                <p>{len(results_df['kernel'].unique())} kernels</p>
            </div>
        </div>

        <div class="chart-container">
            <iframe src="benchmark_results.html"></iframe>
        </div>

        <h2 style="padding: 30px 30px 0 30px; color: #333;">📊 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Matrix Size</th>
                    <th>Kernel</th>
                    <th>Avg Time (ms)</th>
                    <th>{tflop_metric}</th>
                    <th>Bandwidth (GB/s)</th>
                    <th>Speedup</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add table rows
    for _, row in results_df.iterrows():
        if row["kernel"] == "PyTorch":
            kernel_class = "pytorch"
        elif "Naive" in row["kernel"]:
            kernel_class = "naive"
        elif "Shared" in row["kernel"]:
            kernel_class = "shared"
        elif "2D" in row["kernel"]:
            kernel_class = "blocktiling2d"
        elif "Block" in row["kernel"] or "1D" in row["kernel"]:
            kernel_class = "blocktiling"
        elif "Vectorize" in row["kernel"]:
            kernel_class = "vectorize"
        elif "Warptiling" in row["kernel"]:
            kernel_class = "warptiling"
        elif "CUTLASS" in row["kernel"]:
            kernel_class = "cutlass"
        elif "Tensor Core" in row["kernel"]:
            kernel_class = "tensorcore"
        else:
            kernel_class = "coalesced"
        speedup_class = (
            "faster" if row["speedup"] > 1 else "slower" if row["speedup"] < 1 else ""
        )
        speedup_text = (
            f"{row['speedup']:.2f}×" if row["kernel"] != "PyTorch" else "baseline"
        )

        html_content += f"""
                <tr>
                    <td><strong>{row['size']}×{row['size']}</strong></td>
                    <td><span class="kernel-badge badge-{kernel_class}">{row['kernel']}</span></td>
                    <td>{row['avg_time_ms']:.4f}</td>
                    <td><strong>{row['tflops']:.2f}</strong></td>
                    <td>{row['bandwidth_gbps']:.2f}</td>
                    <td><span class="speedup {speedup_class}">{speedup_text}</span></td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>

        <footer>
            <p>Generated by CUDA GEMM Benchmark Suite • Powered by PyTorch & Plotly</p>
        </footer>
    </div>
</body>
</html>
"""

    # Save HTML report
    report_file = output_dir / "index.html"
    with open(report_file, "w") as f:
        f.write(html_content)

    logger.success(f"✅ Saved HTML report to {report_file}")


def run_benchmarks(kernels_to_run: List[str], dtype: str = "float32"):
    """Run benchmarks for specified kernels.

    Args:
        kernels_to_run: List of kernel names to benchmark
        dtype: Data type for input matrices - "float32", "float16", or "bfloat16"
    """
    # Enable TF32 for PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # FP32-only kernels that don't support FP16/BF16
    fp32_only_kernels = {
        "naive",
        "global_mem_coalesce",
        "shared_mem",
        "blocktiling_1d",
        "blocktiling_2d",
        "vectorize",
        "cutlass_fp32",
    }

    # Filter out FP32-only kernels when using FP16/BF16
    if dtype in ["float16", "bfloat16"]:
        incompatible_kernels = [k for k in kernels_to_run if k in fp32_only_kernels]
        if incompatible_kernels:
            logger.warning(
                f"⚠️  The following kernels only support FP32 and will be skipped: {', '.join(incompatible_kernels)}"
            )
            kernels_to_run = [k for k in kernels_to_run if k not in fp32_only_kernels]

    if not kernels_to_run:
        logger.error("❌ No compatible kernels to run!")
        return

    logger.info("🎯 Running GEMM benchmarks...\n")
    logger.info(f"📋 Kernels selected: {', '.join(kernels_to_run)}")
    logger.info(f"🔢 Data type: {dtype}\n")

    # Get GPU memory info
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 Total GPU Memory: {gpu_mem_gb:.2f} GB\n")

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    # Bytes per element for each dtype
    bytes_per_element = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }
    element_size = bytes_per_element[dtype]

    # Test sizes - expanded range up to memory limits
    test_sizes = [64, 96, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192]

    # Store all results
    all_results = []

    for size in test_sizes:
        M = N = K = size
        logger.info(f"{'='*80}")
        logger.info(f"📐 Matrix dimensions: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")

        # Calculate expected memory usage
        memory_per_matrix_gb = (M * K * element_size) / 1e9
        total_memory_gb = 3 * memory_per_matrix_gb  # A, B, C (output is FP32)
        logger.info(
            f"💾 Expected memory usage: {total_memory_gb:.2f} GB ({memory_per_matrix_gb:.2f} GB per matrix)"
        )
        logger.info(f"{'='*80}")

        # Try to allocate tensors, catch OOM errors
        try:
            # Generate random inputs with specified dtype
            a = torch.randn((M, K), device="cuda", dtype=torch_dtype)
            b = torch.randn((K, N), device="cuda", dtype=torch_dtype)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"⚠️  Out of memory! Skipping size {size} and larger sizes."
                )
                logger.warning(
                    f"   Could not allocate {total_memory_gb:.2f} GB for matrices"
                )
                break
            else:
                raise

        # All available kernels (FP32 kernels available for all dtypes)
        all_kernels = [
            ("pytorch", "PyTorch", torch_gemm, "🔵"),
            ("naive", "CUDA Naive", cuda_naive_gemm, "🔴"),
            ("global_mem_coalesce", "CUDA Coalesced", cuda_coalesced_gemm, "🟢"),
            ("shared_mem", "CUDA Shared Mem", cuda_shared_mem_gemm, "🟣"),
            ("blocktiling_1d", "CUDA 1D Block Tiling", cuda_blocktiling_1d_gemm, "🟠"),
            ("blocktiling_2d", "CUDA 2D Block Tiling", cuda_blocktiling_2d_gemm, "🔷"),
            ("vectorize", "CUDA Vectorize", cuda_vectorize_gemm, "💫"),
            ("warptiling", "CUDA Warptiling", cuda_warptiling_gemm, "⚡"),
            ("cutlass_fp32", "CUTLASS (FP32)", cuda_cutlass_fp32_gemm, "🚀"),
        ]

        # Add Tensor Core kernels only for FP16/BF16
        if dtype == "float16":
            all_kernels.extend(
                [
                    (
                        "tensorcore_naive_fp16",
                        "CUDA Tensor Core Naive (FP16)",
                        cuda_tensorcore_naive_fp16_gemm,
                        "🟢",
                    ),
                    (
                        "tensorcore_fp16",
                        "CUDA Tensor Core Warptiled (FP16)",
                        cuda_tensorcore_fp16_gemm,
                        "🚀",
                    ),
                    (
                        "tensorcore_db_fp16",
                        "CUDA Tensor Core Double Buffered (FP16)",
                        cuda_tensorcore_db_fp16_gemm,
                        "🔥",
                    ),
                    (
                        "tensorcore_async_fp16",
                        "CUDA Tensor Core Async (FP16)",
                        cuda_tensorcore_async_fp16_gemm,
                        "💨",
                    ),
                    (
                        "cutlass_fp16",
                        "CUTLASS (FP16)",
                        cuda_cutlass_fp16_gemm,
                        "⚡",
                    ),
                ]
            )
        elif dtype == "bfloat16":
            bf16_kernels = [
                (
                    "tensorcore_naive_bf16",
                    "CUDA Tensor Core Naive (BF16)",
                    cuda_tensorcore_naive_bf16_gemm,
                    "🟢",
                ),
                (
                    "tensorcore_bf16",
                    "CUDA Tensor Core Warptiled (BF16)",
                    cuda_tensorcore_bf16_gemm,
                    "🚀",
                ),
                (
                    "tensorcore_db_bf16",
                    "CUDA Tensor Core Double Buffered (BF16)",
                    cuda_tensorcore_db_bf16_gemm,
                    "🔥",
                ),
                (
                    "tensorcore_async_bf16",
                    "CUDA Tensor Core Async (BF16)",
                    cuda_tensorcore_async_bf16_gemm,
                    "💨",
                ),
                (
                    "cutlass_bf16",
                    "CUTLASS (BF16)",
                    cuda_cutlass_bf16_gemm,
                    "⚡",
                ),
            ]

            # Only add Hopper kernels if they're available (requires SM90+)
            if hasattr(cuda_kernels, "sgemm_cutlass_hopper_bf16"):
                hopper_kernels = [
                    (
                        "cutlass_hopper_bf16",
                        "CUTLASS Hopper (BF16) [Default]",
                        cuda_cutlass_hopper_bf16_gemm,
                        "🔮",
                    ),
                    (
                        "cutlass_hopper_bf16_tma_warp_specialized_auto",
                        "CUTLASS Hopper TMA WarpSpec Auto",
                        cuda_cutlass_hopper_bf16_tma_warp_specialized_auto_gemm,
                        "🌀",
                    ),
                    (
                        "cutlass_hopper_bf16_tma_warp_specialized_constant",
                        "CUTLASS Hopper TMA WarpSpec Const",
                        cuda_cutlass_hopper_bf16_tma_warp_specialized_constant_gemm,
                        "🌀",
                    ),
                    (
                        "cutlass_hopper_bf16_tma_warp_specialized_persistent_constant",
                        "CUTLASS Hopper TMA Persistent Const",
                        cuda_cutlass_hopper_bf16_tma_warp_specialized_persistent_constant_gemm,
                        "🔄",
                    ),
                    (
                        "cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant",
                        "CUTLASS Hopper TMA Pingpong Const",
                        cuda_cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant_gemm,
                        "🏓",
                    ),
                    (
                        "cutlass_hopper_bf16_tma_warp_specialized_streamk_constant",
                        "CUTLASS Hopper TMA Stream-K Const",
                        cuda_cutlass_hopper_bf16_tma_warp_specialized_streamk_constant_gemm,
                        "🌊",
                    ),
                ]
                bf16_kernels.extend(hopper_kernels)

            all_kernels.extend(bf16_kernels)

        # Filter kernels based on user selection
        kernels = [
            (display_name, kernel_fn, emoji)
            for kernel_id, display_name, kernel_fn, emoji in all_kernels
            if kernel_id in kernels_to_run
        ]

        # Warn if user requested kernels that are not available
        available_kernel_ids = {kernel_id for kernel_id, _, _, _ in all_kernels}
        unavailable_kernels = set(kernels_to_run) - available_kernel_ids
        if unavailable_kernels:
            for kernel_id in unavailable_kernels:
                if kernel_id.startswith("cutlass_hopper_"):
                    logger.warning(
                        f"⚠️  Skipping {kernel_id}: Requires Hopper GPU (SM90+). "
                        f"Your GPU: {torch.cuda.get_device_name(0)}"
                    )
                else:
                    logger.warning(f"⚠️  Skipping {kernel_id}: Not available for this configuration")

        size_results = {}

        for kernel_name, kernel_fn, emoji in kernels:
            logger.info(f"\n{emoji} Benchmarking {kernel_name}...")
            try:
                avg_ms, min_ms, max_ms = benchmark_kernel(kernel_fn, a, b)
                tflops, bandwidth = calculate_metrics(M, N, K, avg_ms, element_size)

                size_results[kernel_name] = {
                    "avg_time_ms": avg_ms,
                    "min_time_ms": min_ms,
                    "max_time_ms": max_ms,
                    "tflops": tflops,
                    "bandwidth_gbps": bandwidth,
                }

                logger.info(
                    f"   ⏱️  Time: {avg_ms:.4f} ms (min: {min_ms:.4f}, max: {max_ms:.4f})"
                )
                logger.success(f"   💪 Performance: {tflops:.2f} TFLOPS")
                logger.success(f"   🌊 Bandwidth: {bandwidth:.2f} GB/s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"   ❌ Out of memory during {kernel_name} benchmark!")
                    torch.cuda.empty_cache()
                    continue
                elif "must be multiple of" in str(e) or "must be power of 2" in str(e):
                    logger.warning(
                        f"   ⚠️  Skipping {kernel_name}: incompatible matrix size ({size})"
                    )
                    logger.warning(f"      {str(e)}")
                    continue
                else:
                    raise

        # Skip comparison if not all kernels completed
        if len(size_results) == 0:
            logger.warning("⚠️  No kernels completed for this size, skipping...\n")
            # Clean up and continue
            del a, b
            torch.cuda.empty_cache()
            continue

        # Calculate speedups
        if "PyTorch" not in size_results:
            logger.warning("⚠️  PyTorch baseline missing, skipping comparison\n")
            # Clean up and continue
            del a, b
            torch.cuda.empty_cache()
            continue

        baseline_time = size_results["PyTorch"]["avg_time_ms"]

        logger.info(f"\n{'─'*80}")
        logger.info("📊 Comparison (baseline: PyTorch)")
        logger.info(f"{'─'*80}")

        # Iterate over kernels that were actually run
        for kernel_name in size_results.keys():
            if kernel_name not in size_results:
                logger.warning(f"{kernel_name:20s}: ❌ Failed (OOM)")
                continue

            result = size_results[kernel_name]
            speedup = baseline_time / result["avg_time_ms"]
            result["speedup"] = speedup

            all_results.append(
                {
                    "size": size,
                    "kernel": kernel_name,
                    "avg_time_ms": result["avg_time_ms"],
                    "tflops": result["tflops"],
                    "bandwidth_gbps": result["bandwidth_gbps"],
                    "speedup": speedup,
                }
            )

            if kernel_name == "PyTorch":
                logger.info(f"{kernel_name:20s}: 1.00× (baseline) 🎯")
            else:
                emoji = "🏆" if speedup > 1 else "🐢"
                logger.info(f"{kernel_name:20s}: {speedup:.2f}× {emoji}")

        logger.info("\n")

        # Clean up GPU memory after each size
        del a, b
        torch.cuda.empty_cache()

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Print summary table
    if not results_df.empty:
        logger.info("\n" + "="*130)
        logger.info("📊 BENCHMARK SUMMARY - All Results Ranked by Performance")
        logger.info("="*130 + "\n")

        # Sort by size, then by speedup (descending) within each size
        results_sorted = results_df.sort_values(['size', 'speedup'], ascending=[True, False])

        # Print table header
        logger.info(f"{'Size':<10} {'Kernel':<45} {'Time (ms)':>12} {'TFLOPS':>10} {'Bandwidth (GB/s)':>18} {'Speedup':>12} {'Rank':>6}")
        logger.info("-"*130)

        # Group by size and print results
        current_size = None
        for _, row in results_sorted.iterrows():
            size = row['size']
            kernel = row['kernel']

            # Add separator between different sizes
            if current_size is not None and current_size != size:
                logger.info("-"*130)
            current_size = size

            # Calculate rank within this size group
            size_group = results_sorted[results_sorted['size'] == size]
            rank = (size_group['speedup'] > row['speedup']).sum() + 1

            speedup_str = f"{row['speedup']:.2f}×"

            # Emoji based on speedup
            if abs(row['speedup'] - 1.0) < 0.01:
                emoji = "🎯"  # PyTorch baseline
            elif row['speedup'] > 1.0:
                if rank == 1:
                    emoji = "🥇"  # Best for this size
                elif rank == 2:
                    emoji = "🥈"  # Second best
                elif rank == 3:
                    emoji = "🥉"  # Third best
                else:
                    emoji = "🏆"  # Faster than baseline
            else:
                emoji = "🐢"  # Slower than baseline

            logger.info(
                f"{size:<10} {kernel:<45} {row['avg_time_ms']:>12.4f} {row['tflops']:>10.2f} {row['bandwidth_gbps']:>18.2f} {speedup_str:>11} {emoji} #{rank}"
            )

        logger.info("="*130 + "\n")

        # Print overall best performing kernel (averaged across all sizes)
        overall_summary = results_df.groupby('kernel').agg({
            'speedup': 'mean',
            'tflops': 'mean'
        }).round(4).sort_values('speedup', ascending=False)

        best_kernel = overall_summary.index[0]
        best_speedup = overall_summary.iloc[0]['speedup']
        best_tflops = overall_summary.iloc[0]['tflops']

        logger.success(f"🏆 Overall Best Kernel: {best_kernel}")
        logger.success(f"   Average Speedup: {best_speedup:.2f}× faster than PyTorch")
        logger.success(f"   Average Performance: {best_tflops:.2f} TFLOPS")
        logger.info("")

    # Create output directory
    output_dir = Path(__file__).parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)

    # Save results to CSV
    csv_file = output_dir / "results.csv"
    results_df.to_csv(csv_file, index=False)
    logger.success(f"💾 Saved results to {csv_file}")

    # Create visualizations
    create_visualization(results_df, output_dir, dtype)

    # Create HTML report
    create_html_report(results_df, output_dir, dtype)

    # Open the report in browser
    report_path = output_dir / "index.html"
    logger.success(f"\n🎉 Benchmark complete! Opening {report_path} in browser...")

    # Open in default browser
    webbrowser.open(f"file://{report_path.absolute()}")

    logger.info(f"📂 Results saved to: {output_dir}")
    logger.info(f"   • HTML Report: {report_path}")
    logger.info(f"   • Interactive Charts: {output_dir / 'benchmark_results.html'}")
    logger.info(f"   • CSV Data: {output_dir / 'results.csv'}")


@click.command()
@click.option(
    "--kernels",
    "-k",
    multiple=True,
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
    help="Specify which kernels to benchmark. Can be used multiple times.",
)
@click.option(
    "--dtype",
    "-d",
    type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
    default="float32",
    help="Data type for input matrices (default: float32). Tensor Core kernels require float16 or bfloat16.",
)
def main(kernels, dtype):
    """Benchmark CUDA GEMM kernels against PyTorch.

    Examples:
        # Run all FP32 kernels (default)
        python benchmark.py

        # Run all FP16-compatible kernels (auto-filters to PyTorch, warptiling, tensorcore)
        python benchmark.py -d float16

        # Run all BF16-compatible kernels (auto-filters to PyTorch, warptiling, tensorcore)
        python benchmark.py -d bfloat16

        # Run specific kernels with FP16
        python benchmark.py -d float16 -k pytorch -k tensorcore_fp16

        # Run only PyTorch and naive kernels (FP32)
        python benchmark.py -k pytorch -k naive

        # Run warptiling kernel with all dtypes
        python benchmark.py -k pytorch -k warptiling
        python benchmark.py -d float16 -k pytorch -k warptiling
    """
    # If no kernels specified, run all available for the dtype
    if not kernels:
        # PyTorch and warptiling support all dtypes
        kernels_to_run = ["pytorch", "warptiling"]

        # FP32-only kernels (don't support FP16/BF16)
        if dtype == "float32":
            kernels_to_run.extend(
                [
                    "naive",
                    "global_mem_coalesce",
                    "shared_mem",
                    "blocktiling_1d",
                    "blocktiling_2d",
                    "vectorize",
                    "cutlass_fp32",
                ]
            )

        # Add Tensor Core and CUTLASS kernels for FP16/BF16
        if dtype == "float16":
            kernels_to_run.extend(
                [
                    "tensorcore_naive_fp16",
                    "tensorcore_fp16",
                    "tensorcore_db_fp16",
                    "tensorcore_async_fp16",
                    "cutlass_fp16",
                ]
            )
        elif dtype == "bfloat16":
            bf16_kernels_list = [
                "tensorcore_naive_bf16",
                "tensorcore_bf16",
                "tensorcore_db_bf16",
                "tensorcore_async_bf16",
                "cutlass_bf16",
            ]
            # Only add Hopper kernels if they're available (requires SM90+)
            if hasattr(cuda_kernels, "sgemm_cutlass_hopper_bf16"):
                bf16_kernels_list.extend([
                    "cutlass_hopper_bf16",
                    "cutlass_hopper_bf16_tma_warp_specialized_auto",
                    "cutlass_hopper_bf16_tma_warp_specialized_constant",
                    "cutlass_hopper_bf16_tma_warp_specialized_persistent_constant",
                    "cutlass_hopper_bf16_tma_warp_specialized_pingpong_constant",
                    "cutlass_hopper_bf16_tma_warp_specialized_streamk_constant",
                ])

            kernels_to_run.extend(bf16_kernels_list)
    else:
        kernels_to_run = list(kernels)

    run_benchmarks(kernels_to_run, dtype)


if __name__ == "__main__":
    main()
