// test_gemm_cutlass_blackwell.cpp
// Tests for CUTLASS Blackwell GEMM kernel (BF16 and FP16)
// Uses CUTLASS 3.x Collective Builder API for SM100 (Blackwell)

#define CATCH_CONFIG_MAIN
#include "../../third-party/catch.hpp"
#include "../gemm_kernels.cuh"

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Tolerance for numerical comparison
constexpr float TOLERANCE_BF16 = 1e-2f; // BF16 has lower precision
constexpr float TOLERANCE_FP16 = 1e-3f; // FP16 has better precision

// Helper function to check if tensors are close
bool tensors_are_close(const torch::Tensor &a, const torch::Tensor &b, float tol)
{
    auto diff = (a - b).abs();
    auto max_diff = diff.max().item<float>();
    return max_diff < tol;
}

TEST_CASE("SGEMM CUTLASS Blackwell - Architecture check", "[cutlass_blackwell]")
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Blackwell requires SM 10.0+ (compute capability 10.0 or greater)
    if (prop.major < 10) {
        WARN("Blackwell tests require SM 10.0+ (Blackwell architecture), skipping tests on SM "
             << prop.major << "." << prop.minor);
        REQUIRE(true); // Skip test gracefully
    } else {
        REQUIRE(prop.major >= 10);
    }
}

// BF16 Tests
TEST_CASE("SGEMM CUTLASS Blackwell BF16 - Basic functionality", "[cutlass_blackwell][bf16]")
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        WARN("Skipping BF16 test on SM " << prop.major << "." << prop.minor);
        return;
    }

    torch::manual_seed(42);

    SECTION("Small matrix - 256x256")
    {
        int M = 256, N = 256, K = 256;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_blackwell_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("Medium matrix - 512x512")
    {
        int M = 512, N = 512, K = 512;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_blackwell_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("Large matrix - 1024x1024")
    {
        int M = 1024, N = 1024, K = 1024;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_blackwell_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("Very large matrix - 2048x2048")
    {
        int M = 2048, N = 2048, K = 2048;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_blackwell_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("Non-square matrix - 512x1024x768")
    {
        int M = 512, N = 1024, K = 768;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_blackwell_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }
}

// FP16 Tests
TEST_CASE("SGEMM CUTLASS Blackwell FP16 - Basic functionality", "[cutlass_blackwell][fp16]")
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        WARN("Skipping FP16 test on SM " << prop.major << "." << prop.minor);
        return;
    }

    torch::manual_seed(42);

    SECTION("Small matrix - 256x256")
    {
        int M = 256, N = 256, K = 256;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp16);

        sgemm_cutlass_blackwell_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("Medium matrix - 512x512")
    {
        int M = 512, N = 512, K = 512;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp16);

        sgemm_cutlass_blackwell_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("Large matrix - 1024x1024")
    {
        int M = 1024, N = 1024, K = 1024;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp16);

        sgemm_cutlass_blackwell_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("Non-square matrix - 512x1024x768")
    {
        int M = 512, N = 1024, K = 768;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp16);

        sgemm_cutlass_blackwell_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }
}

// BF16 Kernel Variant Tests
TEST_CASE("SGEMM CUTLASS Blackwell BF16 - Kernel variants", "[cutlass_blackwell][bf16][variants]")
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        WARN("Skipping BF16 variants test on SM " << prop.major << "." << prop.minor);
        return;
    }

    torch::manual_seed(42);
    int M = 512, N = 512, K = 512;
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

    auto A = torch::rand({M, K}, options_bf16);
    auto B = torch::rand({K, N}, options_bf16);
    auto ref = torch::matmul(A, B);

    SECTION("TMA Warp Specialized 2SM - Auto stage count")
    {
        auto C = torch::zeros({M, N}, options_bf16);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_auto(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("TMA Warp Specialized 2SM - Constant stage count")
    {
        auto C = torch::zeros({M, N}, options_bf16);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_constant(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("TMA Warp Specialized 2SM Persistent - Auto stage count")
    {
        auto C = torch::zeros({M, N}, options_bf16);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_persistent_auto(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("TMA Warp Specialized 2SM Persistent - Constant stage count")
    {
        auto C = torch::zeros({M, N}, options_bf16);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_persistent_constant(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("TMA Warp Specialized 2SM Stream-K - Auto stage count")
    {
        auto C = torch::zeros({M, N}, options_bf16);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_streamk_auto(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("TMA Warp Specialized 2SM Stream-K - Constant stage count")
    {
        auto C = torch::zeros({M, N}, options_bf16);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_streamk_constant(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }
}

// FP16 Kernel Variant Tests
TEST_CASE("SGEMM CUTLASS Blackwell FP16 - Kernel variants", "[cutlass_blackwell][fp16][variants]")
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        WARN("Skipping FP16 variants test on SM " << prop.major << "." << prop.minor);
        return;
    }

    torch::manual_seed(42);
    int M = 512, N = 512, K = 512;
    auto options_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

    auto A = torch::rand({M, K}, options_fp16);
    auto B = torch::rand({K, N}, options_fp16);
    auto ref = torch::matmul(A, B);

    SECTION("TMA Warp Specialized 2SM - Auto stage count")
    {
        auto C = torch::zeros({M, N}, options_fp16);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_auto(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("TMA Warp Specialized 2SM - Constant stage count")
    {
        auto C = torch::zeros({M, N}, options_fp16);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_constant(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("TMA Warp Specialized 2SM Persistent - Auto stage count")
    {
        auto C = torch::zeros({M, N}, options_fp16);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_persistent_auto(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("TMA Warp Specialized 2SM Persistent - Constant stage count")
    {
        auto C = torch::zeros({M, N}, options_fp16);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_persistent_constant(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("TMA Warp Specialized 2SM Stream-K - Auto stage count")
    {
        auto C = torch::zeros({M, N}, options_fp16);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_streamk_auto(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("TMA Warp Specialized 2SM Stream-K - Constant stage count")
    {
        auto C = torch::zeros({M, N}, options_fp16);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_streamk_constant(A, B, C);
        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }
}

// Edge case tests
TEST_CASE("SGEMM CUTLASS Blackwell - Edge cases", "[cutlass_blackwell][edge_cases]")
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        WARN("Skipping edge cases test on SM " << prop.major << "." << prop.minor);
        return;
    }

    torch::manual_seed(42);

    SECTION("BF16 - Rectangular tall matrix (2048x512x1024)")
    {
        int M = 2048, N = 512, K = 1024;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_blackwell_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("BF16 - Rectangular wide matrix (512x2048x1024)")
    {
        int M = 512, N = 2048, K = 1024;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_blackwell_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_BF16));
    }

    SECTION("FP16 - Rectangular tall matrix (2048x512x1024)")
    {
        int M = 2048, N = 512, K = 1024;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp16);

        sgemm_cutlass_blackwell_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }

    SECTION("FP16 - Rectangular wide matrix (512x2048x1024)")
    {
        int M = 512, N = 2048, K = 1024;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp16);

        sgemm_cutlass_blackwell_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE_FP16));
    }
}

// Performance comparison test (optional - just verifies all variants produce same results)
TEST_CASE("SGEMM CUTLASS Blackwell - Variant consistency", "[cutlass_blackwell][consistency]")
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 10) {
        WARN("Skipping consistency test on SM " << prop.major << "." << prop.minor);
        return;
    }

    torch::manual_seed(42);
    int M = 1024, N = 1024, K = 1024;

    SECTION("BF16 - All variants produce consistent results")
    {
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);

        auto C_default = torch::zeros({M, N}, options_bf16);
        auto C_auto = torch::zeros({M, N}, options_bf16);
        auto C_constant = torch::zeros({M, N}, options_bf16);
        auto C_persistent = torch::zeros({M, N}, options_bf16);
        auto C_streamk = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_blackwell_bf16(A, B, C_default);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_auto(A, B, C_auto);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_constant(A, B, C_constant);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_persistent_auto(A, B, C_persistent);
        sgemm_cutlass_blackwell_bf16_tma_warp_specialized_2sm_streamk_auto(A, B, C_streamk);

        // All variants should produce nearly identical results
        REQUIRE(tensors_are_close(C_default, C_auto, 1e-5f));
        REQUIRE(tensors_are_close(C_default, C_constant, TOLERANCE_BF16));
        REQUIRE(tensors_are_close(C_default, C_persistent, TOLERANCE_BF16));
        REQUIRE(tensors_are_close(C_default, C_streamk, TOLERANCE_BF16));
    }

    SECTION("FP16 - All variants produce consistent results")
    {
        auto options_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);

        auto C_default = torch::zeros({M, N}, options_fp16);
        auto C_auto = torch::zeros({M, N}, options_fp16);
        auto C_constant = torch::zeros({M, N}, options_fp16);
        auto C_persistent = torch::zeros({M, N}, options_fp16);
        auto C_streamk = torch::zeros({M, N}, options_fp16);

        sgemm_cutlass_blackwell_fp16(A, B, C_default);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_auto(A, B, C_auto);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_constant(A, B, C_constant);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_persistent_auto(A, B, C_persistent);
        sgemm_cutlass_blackwell_fp16_tma_warp_specialized_2sm_streamk_auto(A, B, C_streamk);

        // All variants should produce nearly identical results
        REQUIRE(tensors_are_close(C_default, C_auto, 1e-5f));
        REQUIRE(tensors_are_close(C_default, C_constant, TOLERANCE_FP16));
        REQUIRE(tensors_are_close(C_default, C_persistent, TOLERANCE_FP16));
        REQUIRE(tensors_are_close(C_default, C_streamk, TOLERANCE_FP16));
    }
}
