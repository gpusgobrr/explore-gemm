#include <torch/torch.h>
#include <cuda_runtime.h>
#include "gemm_kernels.cuh"

// CUTLASS 3.x includes for Blackwell Collective Builder
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

using namespace cute;

// Blackwell Warp-Specialized GEMM using CUTLASS 3.x Collective Builder API
// Supports both Sm100 (datacenter) and Sm120 (GeForce) architectures

// Enum to select different Blackwell kernel schedules
enum class BlackwellKernelType
{
    TmaWarpSpecializedCooperative,       // TMA with cooperative warp specialization
    TmaWarpSpecializedPersistent,        // TMA with persistent scheduling
    TmaWarpSpecializedStreamK            // TMA with Stream K scheduling
};

// Enum to select stage count strategy
enum class StageCountType
{
    Auto,    // Automatic stage count calculation with carveout
    Constant // Fixed stage count (5)
};

// Helper function to get kernel schedule type
template <BlackwellKernelType KernelType>
constexpr auto get_kernel_schedule()
{
    // Use auto kernel schedule for all variants
    return cutlass::gemm::collective::KernelScheduleAuto{};
}

// Helper function to get epilogue schedule type
template <BlackwellKernelType KernelType>
constexpr auto get_epilogue_schedule()
{
    // Use auto epilogue schedule for Blackwell
    return cutlass::epilogue::collective::EpilogueScheduleAuto{};
}

// Helper function to get tile scheduler type
template <BlackwellKernelType KernelType>
constexpr auto get_tile_scheduler()
{
    if constexpr (KernelType == BlackwellKernelType::TmaWarpSpecializedCooperative)
    {
        return; // void - no tile scheduler
    }
    else if constexpr (KernelType == BlackwellKernelType::TmaWarpSpecializedStreamK)
    {
        return cutlass::gemm::StreamKScheduler{};
    }
    else
    {
        return cutlass::gemm::PersistentScheduler{};
    }
}

// Helper function to get stage count type with epilogue carveout
template <StageCountType StageType, typename EpilogueType, int Stages = 5>
constexpr auto get_stage_count()
{
    if constexpr (StageType == StageCountType::Auto)
    {
        // Use StageCountAutoCarveout with epilogue shared storage size
        return cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename EpilogueType::SharedStorage))>{};
    }
    else
    {
        return cutlass::gemm::collective::StageCount<Stages>{};
    }
}

template <typename ElementType, BlackwellKernelType KernelType, StageCountType StageType>
struct CutlassBlackwellGemmConfig
{
    // Element types
    using ElementA = ElementType;
    using ElementB = ElementType;
    using ElementC = ElementType;
    using ElementD = ElementType;
    using ElementAccumulator = float;

    // Layouts
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    // Alignment (16-byte for TMA)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Tile and cluster configuration for Blackwell
    // Shape of the tile computed by tcgen05 MMA
    static constexpr int TileM = 128;
    static constexpr int TileN = 128;
    static constexpr int TileK = 64;

    using TileShape = Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>; // CTA tile (M, N, K)

    // Cluster shape for Blackwell
    using ClusterShape = Shape<_1, _1, _1>;

    // Select kernel schedule and epilogue schedule
    using KernelSchedule = decltype(get_kernel_schedule<KernelType>());
    using EpilogueSchedule = decltype(get_epilogue_schedule<KernelType>());
    using TileSchedulerType = decltype(get_tile_scheduler<KernelType>());

    // Build epilogue collective first (needed for stage count carveout)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

    // Get stage count with epilogue carveout
    using StageCount = decltype(get_stage_count<StageType, CollectiveEpilogue>());

    // Build mainloop collective
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        StageCount,
        KernelSchedule>::CollectiveOp;

    // Helper to create the appropriate GemmKernel type
    template <typename Scheduler>
    static auto make_gemm_kernel_type()
    {
        if constexpr (std::is_void_v<Scheduler>)
        {
            return cutlass::gemm::kernel::GemmUniversal<
                Shape<int, int, int>,
                CollectiveMainloop,
                CollectiveEpilogue>{};
        }
        else
        {
            return cutlass::gemm::kernel::GemmUniversal<
                Shape<int, int, int>,
                CollectiveMainloop,
                CollectiveEpilogue,
                Scheduler>{};
        }
    }

    // Assemble the kernel - different signature based on whether we have a tile scheduler
    using GemmKernel = decltype(make_gemm_kernel_type<TileSchedulerType>());

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Type aliases for different kernel configurations
// TMA Warp Specialized Cooperative variants
template <typename ElementType>
using TmaWarpSpecializedCooperativeAutoConfig = CutlassBlackwellGemmConfig<ElementType, BlackwellKernelType::TmaWarpSpecializedCooperative, StageCountType::Auto>;

template <typename ElementType>
using TmaWarpSpecializedCooperativeConstantConfig = CutlassBlackwellGemmConfig<ElementType, BlackwellKernelType::TmaWarpSpecializedCooperative, StageCountType::Constant>;

// TMA Warp Specialized Persistent variants
template <typename ElementType>
using TmaWarpSpecializedPersistentAutoConfig = CutlassBlackwellGemmConfig<ElementType, BlackwellKernelType::TmaWarpSpecializedPersistent, StageCountType::Auto>;

template <typename ElementType>
using TmaWarpSpecializedPersistentConstantConfig = CutlassBlackwellGemmConfig<ElementType, BlackwellKernelType::TmaWarpSpecializedPersistent, StageCountType::Constant>;

// TMA Warp Specialized Stream-K variants
template <typename ElementType>
using TmaWarpSpecializedStreamKAutoConfig = CutlassBlackwellGemmConfig<ElementType, BlackwellKernelType::TmaWarpSpecializedStreamK, StageCountType::Auto>;

template <typename ElementType>
using TmaWarpSpecializedStreamKConstantConfig = CutlassBlackwellGemmConfig<ElementType, BlackwellKernelType::TmaWarpSpecializedStreamK, StageCountType::Constant>;

// BF16 type aliases for all 6 variants
using BF16BlackwellTmaWarpSpecializedCooperativeAuto = TmaWarpSpecializedCooperativeAutoConfig<bfloat16_t>;
using BF16BlackwellTmaWarpSpecializedCooperativeConstant = TmaWarpSpecializedCooperativeConstantConfig<bfloat16_t>;
using BF16BlackwellTmaWarpSpecializedPersistentAuto = TmaWarpSpecializedPersistentAutoConfig<bfloat16_t>;
using BF16BlackwellTmaWarpSpecializedPersistentConstant = TmaWarpSpecializedPersistentConstantConfig<bfloat16_t>;
using BF16BlackwellTmaWarpSpecializedStreamKAuto = TmaWarpSpecializedStreamKAutoConfig<bfloat16_t>;
using BF16BlackwellTmaWarpSpecializedStreamKConstant = TmaWarpSpecializedStreamKConstantConfig<bfloat16_t>;

// Helper to check if scheduler is Stream-K
template <typename Scheduler>
struct is_streamk_scheduler : std::false_type {};

template <>
struct is_streamk_scheduler<cutlass::gemm::StreamKScheduler> : std::true_type {};

template <typename Config>
cudaError_t cutlass_blackwell_gemm_launch(
    int M, int N, int K,
    const typename Config::ElementA *d_A, int lda,
    const typename Config::ElementB *d_B, int ldb,
    typename Config::ElementD *d_D, int ldd,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    typename Config::Gemm gemm_op;

    // Problem size (non-batched GEMM)
    auto problem_shape = make_shape(M, N, K);

    // Stride types for row-major layouts
    using StrideA = typename Config::GemmKernel::StrideA;
    using StrideB = typename Config::GemmKernel::StrideB;
    using StrideC = typename Config::GemmKernel::StrideC;
    using StrideD = typename Config::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {K, N, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // Hardware info with cluster shape configuration
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // Set cluster shape (for 2SM operation, M dimension should be multiple of 2)
    // hw_info.cluster_shape = dim3(4, 4, 1);
    // hw_info.cluster_shape_fallback = dim3(2, 1, 1);

    // Hard-coded alpha = 1.0, beta = 0.0
    float alpha = 1.0f;
    float beta = 0.0f;

    // Create arguments - different for Stream-K vs other schedulers
    typename Config::Gemm::Arguments args = [&]() {
        if constexpr (is_streamk_scheduler<typename Config::TileSchedulerType>::value)
        {
            // Stream-K scheduler requires additional arguments
            using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
            using ReductionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;

            // Stream-K decomposition mode (heuristic for automatic selection)
            DecompositionMode decomp = DecompositionMode::Heuristic;
            ReductionMode reduction = ReductionMode::Deterministic;

            // Number of splits (1 for default Stream-K behavior)
            int splits = 1;

            // Scheduler arguments: splits, raster_order, swizzle, decomposition_mode, reduction_mode
            typename Config::GemmKernel::TileScheduler::Arguments scheduler_args{};
            scheduler_args.splits = splits;
            scheduler_args.max_swizzle_size = 1;
            scheduler_args.decomposition_mode = decomp;
            scheduler_args.reduction_mode = reduction;

            return typename Config::Gemm::Arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                problem_shape,
                {d_A, stride_A, d_B, stride_B},                // Mainloop args
                {{alpha, beta}, d_D, stride_C, d_D, stride_D}, // Epilogue args
                hw_info,
                scheduler_args                                  // Stream-K scheduler args
            };
        }
        else if constexpr (std::is_void_v<typename Config::TileSchedulerType>)
        {
            // No tile scheduler (basic TMA Warp Specialized 2SM)
            return typename Config::Gemm::Arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                problem_shape,
                {d_A, stride_A, d_B, stride_B},                // Mainloop args
                {{alpha, beta}, d_D, stride_C, d_D, stride_D}, // Epilogue args
                hw_info
            };
        }
        else
        {
            // Persistent scheduler (no additional args needed beyond hw_info)
            return typename Config::Gemm::Arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                problem_shape,
                {d_A, stride_A, d_B, stride_B},                // Mainloop args
                {{alpha, beta}, d_D, stride_C, d_D, stride_D}, // Epilogue args
                hw_info
            };
        }
    }();

    // Check if the problem size is supported
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorNotSupported;
    }

    // Initialize the kernel
    size_t workspace_size = Config::Gemm::get_workspace_size(args);
    void *workspace = nullptr;

    if (workspace_size > 0)
    {
        cudaError_t result = cudaMalloc(&workspace, workspace_size);
        if (result != cudaSuccess)
            return result;
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "CUTLASS Blackwell GEMM initialize failed: "
                  << cutlass::cutlassGetStatusString(status)
                  << " at line " << __LINE__ << std::endl;
        if (workspace)
            cudaFree(workspace);
        return cudaErrorUnknown;
    }

    // Run the kernel
    status = gemm_op.run(stream);

    // Free workspace
    if (workspace)
        cudaFree(workspace);

    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "CUTLASS Blackwell GEMM run failed: "
                  << cutlass::cutlassGetStatusString(status)
                  << " at line " << __LINE__ << std::endl;
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

template <typename Config, typename TorchType>
void cutlass_blackwell_gemm_pytorch_wrapper(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    const char *dtype_name,
    const at::ScalarType expected_type)
{
    // Validate input tensors
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == expected_type, "Matrix A must be ", dtype_name);
    TORCH_CHECK(matrix_b.scalar_type() == expected_type, "Matrix B must be ", dtype_name);

    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b.is_contiguous(),
                "Input tensors must be contiguous for alignment requirements");
    TORCH_CHECK(output_matrix.is_contiguous(), "Output tensor must be contiguous");

    // Extract dimensions
    const int M = static_cast<int>(matrix_a.size(0));
    const int K = static_cast<int>(matrix_a.size(1));
    const int N = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == K, "Matrix dimension mismatch");
    TORCH_CHECK(output_matrix.size(0) == M && output_matrix.size(1) == N,
                "Output matrix has wrong shape");

    // Check alignment requirements (16-byte alignment for TMA)
    TORCH_CHECK(reinterpret_cast<uintptr_t>(matrix_a.data_ptr()) % 16 == 0,
                "Matrix A must be 16-byte aligned for Blackwell TMA");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(matrix_b.data_ptr()) % 16 == 0,
                "Matrix B must be 16-byte aligned for Blackwell TMA");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output_matrix.data_ptr()) % 16 == 0,
                "Output matrix must be 16-byte aligned for Blackwell TMA");

    // Get device pointers
    const auto *d_A =
        reinterpret_cast<const typename Config::ElementA *>(matrix_a.data_ptr<TorchType>());
    const auto *d_B =
        reinterpret_cast<const typename Config::ElementB *>(matrix_b.data_ptr<TorchType>());
    auto *d_D = reinterpret_cast<typename Config::ElementD *>(output_matrix.data_ptr<TorchType>());

    int lda = K;
    int ldb = N;
    int ldd = N;

    cudaStream_t stream = nullptr;

    // Launch CUTLASS Blackwell GEMM (alpha=1.0, beta=0.0 hard-coded)
    const cudaError_t err = cutlass_blackwell_gemm_launch<Config>(
        M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);

    // Synchronize to catch any kernel launch errors
    if (err == cudaSuccess) {
        cudaError_t sync_err = cudaDeviceSynchronize();
        TORCH_CHECK(sync_err == cudaSuccess,
                    "CUTLASS Blackwell GEMM (", dtype_name, ") kernel execution failed: ",
                    cudaGetErrorString(sync_err));
    }

    std::cout << "CUTLASS Blackwell GEMM (" << dtype_name << ") launched: "
              << "M=" << M << ", N=" << N << ", K=" << K << "err: "<< err << std::endl;

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS Blackwell GEMM (", dtype_name, ") launch failed: ", cudaGetErrorString(err));
}

// BF16 launchers - Default variant (uses Cooperative Auto as the default)
void sgemm_cutlass_blackwell_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_blackwell_gemm_pytorch_wrapper<BF16BlackwellTmaWarpSpecializedCooperativeAuto, at::BFloat16>(
        matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}

// BF16 launchers - TMA Warp Specialized Cooperative variants
void sgemm_cutlass_blackwell_bf16_tma_warp_specialized_cooperative_auto(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_blackwell_gemm_pytorch_wrapper<BF16BlackwellTmaWarpSpecializedCooperativeAuto, at::BFloat16>(
        matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}

void sgemm_cutlass_blackwell_bf16_tma_warp_specialized_cooperative_constant(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_blackwell_gemm_pytorch_wrapper<BF16BlackwellTmaWarpSpecializedCooperativeConstant, at::BFloat16>(
        matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}

// BF16 launchers - TMA Warp Specialized Persistent variants
void sgemm_cutlass_blackwell_bf16_tma_warp_specialized_persistent_auto(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_blackwell_gemm_pytorch_wrapper<BF16BlackwellTmaWarpSpecializedPersistentAuto, at::BFloat16>(
        matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}

void sgemm_cutlass_blackwell_bf16_tma_warp_specialized_persistent_constant(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_blackwell_gemm_pytorch_wrapper<BF16BlackwellTmaWarpSpecializedPersistentConstant, at::BFloat16>(
        matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}

// BF16 launchers - TMA Warp Specialized Stream-K variants
void sgemm_cutlass_blackwell_bf16_tma_warp_specialized_streamk_auto(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_blackwell_gemm_pytorch_wrapper<BF16BlackwellTmaWarpSpecializedStreamKAuto, at::BFloat16>(
        matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}

void sgemm_cutlass_blackwell_bf16_tma_warp_specialized_streamk_constant(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_blackwell_gemm_pytorch_wrapper<BF16BlackwellTmaWarpSpecializedStreamKConstant, at::BFloat16>(
        matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}

