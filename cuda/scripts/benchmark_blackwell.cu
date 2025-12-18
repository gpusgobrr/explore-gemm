#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "common.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using ElementA = half_t;                                                // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;                              // Layout type for A matrix operand
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB = half_t;                                                // Element type for B matrix operand
using LayoutB = cutlass::layout::ColumnMajor;                           // Layout type for B matrix operand
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementC = half_t;                                                // Element type for C and D matrix operands
using LayoutC = cutlass::layout::ColumnMajor;                           // Layout type for C and D matrix operands
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator = float;                     // Element type for internal accumulation
using ArchTag = cutlass::arch::Sm100;                 // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag

// MMA and Cluster Tile Shapes
// Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape % 2 == 0
using MmaTileShape_MNK = Shape<_256, _128, _64>;
// Shape of the cluster set to <int,int,_1> to indicate dynamic cluster shape
using ClusterShape_MNK = Shape<int, int, _1>;
// When dynamic cluster is used, KernelScheduleAuto always selects mainloop dispatch policy that
// lowers to tcgen05 MMA cta_group = 1 as we don't know if the dynamic cluster M dimension will be a multiple of 2
// To use tcgen05 MMA cta_group = 2, users must explicitly use 2sm builder schedules
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
using ReductionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;

// Command line options parsing
struct Options
{

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  int preferred_cluster_m, preferred_cluster_n;
  int fallback_cluster_m, fallback_cluster_n;
  RasterOrderOptions raster;
  DecompositionMode decomp;
  ReductionMode reduction;
  int swizzle;
  int splits;
  bool csv;

  std::unordered_map<DecompositionMode, std::vector<std::string>> dec_mappings = {
      {DecompositionMode::Heuristic, {"heuristic", "Heuristic", "h", "H"}},
      {DecompositionMode::StreamK, {"streamk", "StreamK", "stream-k", "Stream-K", "stk", "StK"}},
      {DecompositionMode::SplitK, {"splitk", "SplitK", "split-k", "Split-K", "spk", "SpK"}},
      {DecompositionMode::DataParallel, {"dataparallel", "DataParallel", "data-parallel", "dp", "DP"}}};

  std::unordered_map<ReductionMode, std::vector<std::string>> red_mappings = {
      {ReductionMode::Deterministic, {"deterministic", "Deterministic", "d", "D"}},
      {ReductionMode::Nondeterministic, {"nondeterministic", "Nondeterministic", "n", "N"}}};

  Options() : help(false),
              m(2048), n(2048), k(2048),
              alpha(1.f), beta(0.f),
              iterations(100),
              preferred_cluster_m(4), preferred_cluster_n(4),
              fallback_cluster_m(2), fallback_cluster_n(1),
              raster(RasterOrderOptions::Heuristic),
              swizzle(1),
              decomp(DecompositionMode::Heuristic),
              reduction(ReductionMode::Deterministic),
              splits(1),
              csv(false)
  {
  }

  // Parses the command line
  void parse(int argc, char const **args)
  {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help"))
    {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n')
    {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm')
    {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h')
    {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);
    cmd.get_cmd_line_argument("splits", splits, 1);
    cmd.get_cmd_line_argument("preferred_cluster_m", preferred_cluster_m, 4);
    cmd.get_cmd_line_argument("preferred_cluster_n", preferred_cluster_n, 4);
    cmd.get_cmd_line_argument("fallback_cluster_m", fallback_cluster_m, 2);
    cmd.get_cmd_line_argument("fallback_cluster_n", fallback_cluster_n, 1);

    // Parse decomposition mode
    std::string decomposition;
    cmd.get_cmd_line_argument("decomposition", decomposition);
    if (!decomposition.empty())
    {
      bool found = parse_from_options_map(decomposition, dec_mappings, decomp);
      if (!found)
      {
        std::cout << "--decomposition must be one of: heuristic, streamk, splitk, dataparallel" << std::endl;
        help = true;
        return;
      }
    }

    // Parse reduction mode
    std::string red_mode;
    cmd.get_cmd_line_argument("reduction", red_mode);
    if (!red_mode.empty())
    {
      bool found = parse_from_options_map(red_mode, red_mappings, reduction);
      if (!found)
      {
        std::cout << "--reduction must be one of: deterministic, nondeterministic" << std::endl;
        help = true;
        return;
      }
    }

    cmd.get_cmd_line_argument("csv", csv, false);
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const
  {

    out << "benchmark_blackwell\n\n"
        << "  Blackwell GEMM benchmark for various tile scheduler configurations.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n\n"
        << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
        << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
        << "  --preferred_cluster_m=<int> Preferred cluster shape M dimension (default: 4)\n"
        << "  --preferred_cluster_n=<int> Preferred cluster shape N dimension (default: 4)\n"
        << "  --fallback_cluster_m=<int>  Fallback cluster shape M dimension (default: 2)\n"
        << "  --fallback_cluster_n=<int>  Fallback cluster shape N dimension (default: 1)\n\n"
        << "  --decomposition=<string>    Decomposition Mode (heuristic, streamk, splitk, dataparallel)\n\n"
        << "  --reduction=<string>        Reduction Mode (deterministic, nondeterministic)\n\n"
        << "  --splits=<int>              Number of K split (Only used in splitk mode)\n\n"
        << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
        << "  --csv                       Output results in CSV format\n\n";

    out
        << "\n\nExamples:\n\n"
        << "$ ./benchmark_blackwell --m=1024 --n=1024 --k=4096 --decomposition=streamk --reduction=deterministic --raster=H \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options)
{

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, 1},
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}};

  arguments.hw_info.cluster_shape = dim3(options.preferred_cluster_m, options.preferred_cluster_n, 1);
  arguments.hw_info.cluster_shape_fallback = dim3(options.fallback_cluster_m, options.fallback_cluster_n, 1);

  arguments.scheduler.splits = options.splits;
  arguments.scheduler.decomposition_mode = options.decomp;
  arguments.scheduler.reduction_mode = options.reduction;
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

bool verify(const Options &options)
{
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
      {options.m, options.n, options.k},
      ElementAccumulator(options.alpha),
      ref_A,
      ref_B,
      ElementAccumulator(options.beta),
      ref_C,
      ref_D);

  // Wait for kernel to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  BenchmarkResult result;
  result.passed = verify(options);

  if (!options.csv)
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed)
  {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter)
    {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN)
    {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM)
    {
      raster = "Along M";
    }

    std::string decomp = "Heuristic";
    if (options.decomp == DecompositionMode::StreamK)
    {
      decomp = "StreamK";
    }
    else if (options.decomp == DecompositionMode::SplitK)
    {
      decomp = "SplitK";
    }
    else if (options.decomp == DecompositionMode::DataParallel)
    {
      decomp = "DataParallel";
    }

    std::string red = "Deterministic";
    if (options.reduction == ReductionMode::Nondeterministic)
    {
      red = "Nondeterministic";
    }

    const int worktile_count = options.m / cute::get<0>(MmaTileShape_MNK{}) * options.n / cute::get<1>(MmaTileShape_MNK{});

    if (options.csv)
    {
      std::cout << options.m << ',' << options.n << ',' << options.k << ','
                << raster << ',' << options.swizzle << ','
                << decomp << ',' << options.splits << ','
                << red << ','
                << "(" << options.preferred_cluster_m << "," << options.preferred_cluster_n << ")" << ','
                << "(" << options.fallback_cluster_m << "," << options.fallback_cluster_n << ")" << ','
                << result.avg_runtime_ms << "," << result.gflops << ','
                << worktile_count << std::endl;
    }
    else
    {
      std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
      std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
      std::cout << "  Decomposition: " << decomp << ((options.decomp == DecompositionMode::SplitK) ? " with split of " + std::to_string(options.splits) : "") << std::endl;
      std::cout << "  Reduction: " << red << std::endl;
      std::cout << "  Preferred Cluster: (" << options.preferred_cluster_m << ", " << options.preferred_cluster_n << ", 1)" << std::endl;
      std::cout << "  Fallback Cluster: (" << options.fallback_cluster_m << ", " << options.fallback_cluster_n << ", 1)" << std::endl;
      std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
      std::cout << "  GFLOPS: " << result.gflops << std::endl;
      std::cout << "  Worktile Count: " << worktile_count << std::endl;
    }
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args)
{

  // CUTLASS must be compiled with CUDA 12.8 Toolkit or newer to run this example
  // and must have compute capability at least 100.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8))
  {
    std::cerr << "This example requires CUDA 12.8 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 10)
  {
    std::cerr
        << "This example requires a GPU of NVIDIA's Blackwell Architecture or "
        << "later (compute capability 100 or greater).\n";
    return 0;
  }
  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help)
  {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
