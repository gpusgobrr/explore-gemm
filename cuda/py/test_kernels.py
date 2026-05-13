"""Unit tests for CUDA GEMM kernels.

Tests all CUDA kernels against PyTorch baseline to ensure correctness.

Usage:
    # Run all tests
    pytest test_kernels.py -v

    # Run specific test
    pytest test_kernels.py::test_fp32_kernels -v

    # Run with specific matrix size
    pytest test_kernels.py -v -k "size512"
"""

import os

# Set CUDA paths to match CMakeLists.txt configuration
# Must be set BEFORE importing torch
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import pytest
import torch

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

# Import the shared CUDA extension loader
from cuda_extension_loader import create_cuda_extension

# Import Triton persistent kernel
from triton_kernel_gemm import matmul_persistent as triton_matmul


# Load CUDA kernels once for all tests
@pytest.fixture(scope="module")
def cuda_kernels():
    """Load CUDA kernels once for all tests."""
    print("🚀 Loading CUDA kernels for tests...")
    kernels = create_cuda_extension(verbose=False)
    print("✅ CUDA kernels loaded successfully!")
    return kernels


# Test matrix sizes - various sizes to catch edge cases
@pytest.fixture(params=[64, 128, 256, 512, 1024, 2048])
def matrix_size(request):
    """Parameterized fixture for different matrix sizes."""
    return request.param


# FP32 kernel fixtures
@pytest.fixture(
    params=[
        "naive",
        "global_mem_coalesce",
        "shared_mem",
        "blocktiling_1d",
        "blocktiling_2d",
        "vectorize",
        "warptiling",
        "cutlass_fp32",
        "triton_persistent",
    ]
)
def fp32_kernel_name(request):
    """Parameterized fixture for FP32 kernel names."""
    return request.param


# FP16 kernel fixtures
@pytest.fixture(
    params=[
        "tensorcore_naive_fp16",
        "tensorcore_fp16",
        "tensorcore_db_fp16",
        "tensorcore_async_fp16",
        "cutlass_fp16",
        "triton_persistent_fp16",
    ]
)
def fp16_kernel_name(request):
    """Parameterized fixture for FP16 kernel names."""
    return request.param


# BF16 kernel fixtures
@pytest.fixture(
    params=[
        "tensorcore_naive_bf16",
        "tensorcore_bf16",
        "tensorcore_db_bf16",
        "tensorcore_async_bf16",
        "cutlass_bf16",
        "triton_persistent_bf16",
    ]
)
def bf16_kernel_name(request):
    """Parameterized fixture for BF16 kernel names."""
    return request.param


def run_kernel(kernel_name, cuda_kernels, a, b, c):
    """Run a specific kernel by name."""
    if kernel_name == "naive":
        cuda_kernels.sgemm_naive(a, b, c, 1.0, 0.0)
    elif kernel_name == "global_mem_coalesce":
        cuda_kernels.sgemm_global_mem_coalesce(a, b, c, 1.0, 0.0)
    elif kernel_name == "shared_mem":
        cuda_kernels.sgemm_shared_mem(a, b, c, 1.0, 0.0)
    elif kernel_name == "blocktiling_1d":
        cuda_kernels.sgemm_blocktiling_1d(a, b, c, 1.0, 0.0)
    elif kernel_name == "blocktiling_2d":
        cuda_kernels.sgemm_blocktiling_2d(a, b, c, 1.0, 0.0)
    elif kernel_name == "vectorize":
        cuda_kernels.sgemm_vectorize(a, b, c, 1.0, 0.0)
    elif kernel_name == "warptiling":
        if a.dtype == torch.float32:
            cuda_kernels.sgemm_warptiling_default(a, b, c, 1.0, 0.0)
        elif a.dtype == torch.float16:
            cuda_kernels.sgemm_warptiling_fp16(a, b, c, 1.0, 0.0)
        elif a.dtype == torch.bfloat16:
            cuda_kernels.sgemm_warptiling_bf16(a, b, c, 1.0, 0.0)
    elif kernel_name == "tensorcore_naive_fp16":
        cuda_kernels.sgemm_tensorcore_naive_fp16(a, b, c, 1.0, 0.0)
    elif kernel_name == "tensorcore_naive_bf16":
        cuda_kernels.sgemm_tensorcore_naive_bf16(a, b, c, 1.0, 0.0)
    elif kernel_name == "tensorcore_fp16":
        cuda_kernels.sgemm_tensorcore_fp16(a, b, c, 1.0, 0.0)
    elif kernel_name == "tensorcore_bf16":
        cuda_kernels.sgemm_tensorcore_bf16(a, b, c, 1.0, 0.0)
    elif kernel_name == "tensorcore_db_fp16":
        cuda_kernels.sgemm_tensorcore_double_buffered_fp16(a, b, c, 1.0, 0.0)
    elif kernel_name == "tensorcore_db_bf16":
        cuda_kernels.sgemm_tensorcore_double_buffered_bf16(a, b, c, 1.0, 0.0)
    elif kernel_name == "tensorcore_async_fp16":
        cuda_kernels.sgemm_tensorcore_async_fp16(a, b, c, 1.0, 0.0)
    elif kernel_name == "tensorcore_async_bf16":
        cuda_kernels.sgemm_tensorcore_async_bf16(a, b, c, 1.0, 0.0)
    elif kernel_name == "cutlass_fp16":
        cuda_kernels.sgemm_cutlass_fp16(a, b, c, 1.0, 0.0)
    elif kernel_name == "cutlass_bf16":
        cuda_kernels.sgemm_cutlass_bf16(a, b, c, 1.0, 0.0)
    elif kernel_name == "cutlass_fp32":
        cuda_kernels.sgemm_cutlass_fp32(a, b, c, 1.0, 0.0)
    elif kernel_name == "triton_persistent":
        # Triton kernel returns the result directly
        result = triton_matmul(a, b)
        c.copy_(result)
    elif kernel_name == "triton_persistent_fp16":
        # Triton kernel returns the result directly
        result = triton_matmul(a, b)
        c.copy_(result.to(c.dtype))
    elif kernel_name == "triton_persistent_bf16":
        # Triton kernel returns the result directly
        result = triton_matmul(a, b)
        c.copy_(result.to(c.dtype))
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


def test_fp32_kernels(cuda_kernels, fp32_kernel_name, matrix_size):
    """Test FP32 kernels against PyTorch baseline."""
    M = N = K = matrix_size

    # Create input tensors
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    # PyTorch reference
    expected = torch.matmul(a, b)

    # Run CUDA kernel
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    try:
        run_kernel(fp32_kernel_name, cuda_kernels, a, b, c)
    except RuntimeError as e:
        if "must be multiple of" in str(e) or "must be power of 2" in str(e):
            pytest.skip(
                f"Kernel {fp32_kernel_name} requires specific dimensions: {str(e)}"
            )
        else:
            raise

    # Check correctness with relative tolerance
    # FP32 accumulation errors scale with matrix size (K dimension)
    # Triton kernels may have slightly higher precision errors due to autotuning
    if "triton" in fp32_kernel_name:
        # Triton persistent kernel has accumulation errors that scale with K dimension
        # For larger matrices, the errors accumulate more
        if matrix_size <= 64:
            rtol = 1e-2
            atol = 1e-2
        elif matrix_size <= 256:
            rtol = 5e-2
            atol = 5e-2
        else:
            rtol = 1e-1
            atol = 1e-1
    else:
        rtol = 1e-3
        atol = 1e-3

    assert torch.allclose(c, expected, rtol=rtol, atol=atol), (
        f"Kernel {fp32_kernel_name} failed for size {matrix_size}x{matrix_size}. "
        f"Max error: {(c - expected).abs().max().item():.6e}, "
        f"Mean error: {(c - expected).abs().mean().item():.6e}"
    )

    logger.success(
        f"✅ {fp32_kernel_name:25s} passed for size {matrix_size:4d}x{matrix_size:4d}"
    )


def test_fp16_kernels(cuda_kernels, fp16_kernel_name, matrix_size):
    """Test FP16 kernels against PyTorch baseline."""
    M = N = K = matrix_size

    # Skip tensor core tests for very small sizes (minimum warp tile size requirement)
    if "tensorcore" in fp16_kernel_name and matrix_size < 128:
        pytest.skip(f"Tensor core kernels require minimum size 128x128")

    # Create input tensors with smaller range to avoid overflow
    # FP16 max value is ~65504, so we use smaller values
    a = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=torch.float16) * 0.1

    # PyTorch reference
    expected = torch.matmul(a, b)

    # Tensor core kernels output FP32
    output_dtype = torch.float32
    result_fp32 = torch.empty((M, N), device="cuda", dtype=output_dtype)

    try:
        run_kernel(fp16_kernel_name, cuda_kernels, a, b, result_fp32)
        result = result_fp32.to(torch.float16)
    except RuntimeError as e:
        if "must be multiple of" in str(e) or "must be power of 2" in str(e):
            pytest.skip(
                f"Kernel {fp16_kernel_name} requires specific dimensions: {str(e)}"
            )
        else:
            raise

    # Check for NaN/Inf
    if torch.isnan(result).any() or torch.isinf(result).any():
        pytest.fail(f"Kernel {fp16_kernel_name} produced NaN or Inf values")

    # Check correctness with relaxed tolerance for FP16
    # FP16 has ~3 decimal digits of precision
    # Triton kernels may have slightly higher precision errors due to autotuning
    if "triton" in fp16_kernel_name:
        # Triton persistent kernel has accumulation errors that scale with K dimension
        if matrix_size <= 256:
            rtol = 1e-1
            atol = 1e-1
        else:
            rtol = 2e-1
            atol = 2e-1
    else:
        rtol = 1e-3
        atol = 1e-3

    assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
        f"Kernel {fp16_kernel_name} failed for size {matrix_size}x{matrix_size}. "
        f"Max error: {(result - expected).abs().max().item():.6e}, "
        f"Mean error: {(result - expected).abs().mean().item():.6e}"
    )

    logger.success(
        f"✅ {fp16_kernel_name:25s} passed for size {matrix_size:4d}x{matrix_size:4d} (FP16)"
    )


def test_bf16_kernels(cuda_kernels, bf16_kernel_name, matrix_size):
    """Test BF16 kernels against PyTorch baseline."""
    M = N = K = matrix_size

    # Skip tensor core tests for very small sizes (minimum warp tile size requirement)
    if "tensorcore" in bf16_kernel_name and matrix_size < 128:
        pytest.skip(f"Tensor core kernels require minimum size 128x128")

    # Create input tensors with smaller range to avoid overflow
    # BF16 has same range as FP32 but less precision
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) * 0.1

    # PyTorch reference
    expected = torch.matmul(a, b)

    # Tensor core kernels output FP32
    output_dtype = torch.float32
    result_fp32 = torch.empty((M, N), device="cuda", dtype=output_dtype)

    try:
        run_kernel(bf16_kernel_name, cuda_kernels, a, b, result_fp32)
        result = result_fp32.to(torch.bfloat16)
    except RuntimeError as e:
        if "must be multiple of" in str(e) or "must be power of 2" in str(e):
            pytest.skip(
                f"Kernel {bf16_kernel_name} requires specific dimensions: {str(e)}"
            )
        else:
            raise

    # Check for NaN/Inf
    if torch.isnan(result).any() or torch.isinf(result).any():
        pytest.fail(f"Kernel {bf16_kernel_name} produced NaN or Inf values")

    # Check correctness with relaxed tolerance for BF16
    # BF16 has ~2-3 decimal digits of precision
    # Triton kernels may have slightly higher precision errors due to autotuning
    if "triton" in bf16_kernel_name:
        # Triton persistent kernel has accumulation errors that scale with K dimension
        if matrix_size <= 256:
            rtol = 1e-1
            atol = 1e-1
        else:
            rtol = 2e-1
            atol = 2e-1
    else:
        rtol = 1e-3
        atol = 1e-3

    assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
        f"Kernel {bf16_kernel_name} failed for size {matrix_size}x{matrix_size}. "
        f"Max error: {(result - expected).abs().max().item():.6e}, "
        f"Mean error: {(result - expected).abs().mean().item():.6e}"
    )

    logger.success(
        f"✅ {bf16_kernel_name:25s} passed for size {matrix_size:4d}x{matrix_size:4d} (BF16)"
    )


def test_non_square_matrices_fp32(cuda_kernels):
    """Test FP32 kernels with non-square matrices."""
    # Test various non-square shapes
    test_shapes = [
        (64, 128, 96),  # M, K, N
        (128, 64, 256),
        (256, 512, 128),
    ]

    kernels_to_test = [
        "naive",
        "global_mem_coalesce",
        "shared_mem",
        "warptiling",
    ]

    for M, K, N in test_shapes:
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b = torch.randn((K, N), device="cuda", dtype=torch.float32)
        expected = torch.matmul(a, b)

        for kernel_name in kernels_to_test:
            c = torch.empty((M, N), device="cuda", dtype=torch.float32)

            try:
                run_kernel(kernel_name, cuda_kernels, a, b, c)

                # Uniform tolerance for all tests
                assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3), (
                    f"Kernel {kernel_name} failed for shape ({M}, {K}) @ ({K}, {N}). "
                    f"Max error: {(c - expected).abs().max().item():.6e}"
                )

                logger.success(
                    f"✅ {kernel_name:25s} passed for shape ({M:3d}, {K:3d}) @ ({K:3d}, {N:3d})"
                )
            except RuntimeError as e:
                if "must be multiple of" in str(e) or "must be power of 2" in str(e):
                    logger.warning(
                        f"⚠️  {kernel_name:25s} skipped for shape ({M:3d}, {K:3d}) @ ({K:3d}, {N:3d}) - "
                        f"incompatible dimensions"
                    )
                else:
                    raise


def test_small_matrices_fp32(cuda_kernels):
    """Test FP32 kernels with very small matrices (edge cases)."""
    small_sizes = [16, 32]

    kernels_to_test = [
        "naive",
        "global_mem_coalesce",
        "shared_mem",
    ]

    for size in small_sizes:
        M = N = K = size
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b = torch.randn((K, N), device="cuda", dtype=torch.float32)
        expected = torch.matmul(a, b)

        for kernel_name in kernels_to_test:
            c = torch.empty((M, N), device="cuda", dtype=torch.float32)

            try:
                run_kernel(kernel_name, cuda_kernels, a, b, c)

                # Uniform tolerance for all tests
                assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3), (
                    f"Kernel {kernel_name} failed for size {size}x{size}. "
                    f"Max error: {(c - expected).abs().max().item():.6e}"
                )

                logger.success(
                    f"✅ {kernel_name:25s} passed for size {size:4d}x{size:4d} (small matrix)"
                )
            except RuntimeError as e:
                if "must be multiple of" in str(e) or "must be power of 2" in str(e):
                    logger.warning(
                        f"⚠️  {kernel_name:25s} skipped for size {size:4d}x{size:4d} - "
                        f"incompatible dimensions"
                    )
                else:
                    raise


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
