#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <optional>

#include <torch/torch.h>

#include "cutlass/util/command_line.h"
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_scaling_factor.h"
#include "cutlass/gemm/device/gemv_blockscaled.h"
#include "cutlass/gemm/kernel/gemv_blockscaled.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_size.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

template <typename T>
auto make_iterator(T *ptr)
{
  return cute::recast_ptr<T>(ptr);
}

template <typename Gemv_>
struct TestbedPyTorchGemvFp4SFD
{
public:
  using Gemv = Gemv_;

  using ElementA = typename Gemv::ElementA;
  using ElementSFA = typename Gemv::ElementSFA;
  using LayoutA = typename Gemv::LayoutA;

  using ElementB = typename Gemv::ElementB;
  using ElementSFB = typename Gemv::ElementSFB;
  using LayoutB = cutlass::layout::ColumnMajor;

  using ElementC = typename Gemv::ElementC;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementD = typename Gemv::EpilogueOutputOp::ElementD;
  using LayoutD = typename Gemv::EpilogueOutputOp::LayoutOutput;

  using ElementSFD = typename Gemv::EpilogueOutputOp::ElementSFD;
  using LayoutSFD = typename Gemv::EpilogueOutputOp::LayoutSFD;

  using ElementAccumulator = typename Gemv::ElementAccumulator;
  using ElementCompute = typename Gemv::EpilogueOutputOp::ElementCompute;

  static constexpr int kVectorSize = Gemv::EpilogueOutputOp::kVectorSize;

  using Sm1xxBlockScaledOutputConfig =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<kVectorSize,
                                                    cutlass::is_same_v<LayoutSFD, cutlass::layout::RowMajor> ? cute::UMMA::Major::K : cute::UMMA::Major::MN>;
  using Blk_MN_Output = typename Sm1xxBlockScaledOutputConfig::Blk_MN;
  using Blk_SF_Output = typename Sm1xxBlockScaledOutputConfig::Blk_SF;

  using Sm100BlockScaledInputConfig = cutlass::detail::Sm1xxBlockScaledConfig<kVectorSize>;
  using Blk_MN_Input = typename Sm100BlockScaledInputConfig::Blk_MN;
  using Blk_SF_Input = typename Sm100BlockScaledInputConfig::Blk_SF;
  using SfAtom_Input = typename Sm100BlockScaledInputConfig::SfAtom;

public:
  bool run_gemv(
      torch::Tensor A,
      torch::Tensor B,
      torch::Tensor C,
      torch::Tensor D,
      torch::Tensor SFA,
      torch::Tensor SFB,
      torch::Tensor SFD,
      int32_t batch_count,
      ElementCompute alpha,
      ElementCompute beta,
      float epilogue_st,
      bool is_profiling,
      int kIterations)
  {

    if (A.dim() != 2 || B.dim() != 2 || D.dim() != 2)
    {
      printf("Invalid tensor dimensions. Expected 2D tensors.\n");
      return false;
    }

    const int32_t gemm_m = A.size(0) / batch_count;
    const int32_t gemm_k = A.size(1);
    const int32_t gemm_n = 1;

    int k_blks_input = cutlass::ceil_div(gemm_k, cute::size<1>(cute::shape(SfAtom_Input{})));
    int m_blks_input = cutlass::ceil_div(gemm_m, cute::size(Blk_MN_Input{}));
    int n_blks_input = cutlass::ceil_div(gemm_n, cute::size(Blk_MN_Input{}));

    int batch_stride_SFA = m_blks_input * cute::size(Blk_MN_Input{}) * k_blks_input * cute::size(Blk_SF_Input{});
    int batch_stride_SFB = n_blks_input * cute::size(Blk_MN_Input{}) * k_blks_input * cute::size(Blk_SF_Input{});

    using ProblemShapeType = cute::Shape<int, int, int, int>;
    auto problem_shape_MNKL = ProblemShapeType{gemm_m, gemm_n, gemm_k, batch_count};

    auto sfd_layout = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(problem_shape_MNKL);
    auto batch_stride_tuple = cute::stride<2>(sfd_layout);
    int batch_stride_SFD = static_cast<int>(cute::get<1>(batch_stride_tuple));

    Gemv gemv_op;

    // Create TensorRef objects from PyTorch tensor data pointers
    typename Gemv::LayoutA layout_A;
    cutlass::TensorRef<ElementA, LayoutA> ref_A((ElementA *)A.data_ptr(), layout_A(cutlass::MatrixCoord{batch_count * gemm_m, gemm_k}));
    cutlass::TensorRef<ElementD, LayoutD> ref_D((ElementD *)D.data_ptr(), LayoutD(cutlass::MatrixCoord{batch_count * gemm_m, 1}));

    typename Gemv::Arguments arguments{
        cutlass::MatrixCoord{gemm_m, gemm_k},
        batch_count,
        typename Gemv::EpilogueOutputOp::Params{
            ref_D,
            static_cast<ElementSFD*>(SFD.data_ptr()),
            alpha,
            beta,
            epilogue_st,
            batch_stride_SFD,
            gemm_m},
        ref_A,
        (ElementB *)B.data_ptr(),
        (ElementC *)C.data_ptr(),
        (ElementD *)D.data_ptr(),
        (ElementSFA *)SFA.data_ptr(),
        (ElementSFB *)SFB.data_ptr(),
        gemm_k,
        gemm_m * gemm_k,
        gemm_k,
        gemm_m,
        gemm_m,
        batch_stride_SFA,
        batch_stride_SFB,
        batch_stride_SFD};

    cutlass::Status status = gemv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
    {
      printf("can_implement() failed\n");
      return false;
    }

    size_t workspace_size = Gemv::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemv_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess)
    {
      printf("initialize() failed\n");
      return false;
    }

    if (not is_profiling)
    {
      status = gemv_op();
    }
    else
    {
      cudaError_t result;
      cudaEvent_t events[2];

      for (cudaEvent_t &evt : events)
      {
        result = cudaEventCreate(&evt);
        if (result != cudaSuccess)
        {
          std::cerr << "cudaEventCreate failed with error " << cudaGetErrorString(result) << std::endl;
          return false;
        }
      }

      status = gemv_op();
      if (status != cutlass::Status::kSuccess)
      {
        std::cerr << "Device execution failed on warmup." << std::endl;
        return false;
      }

      result = cudaEventRecord(events[0]);
      if (result != cudaSuccess)
      {
        std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }

      for (int iter_i = 0; iter_i < kIterations; ++iter_i)
      {
        status = gemv_op();
        if (status != cutlass::Status::kSuccess)
        {
          std::cerr << "Device execution failed." << std::endl;
          return false;
        }
      }

      result = cudaEventRecord(events[1]);
      if (result != cudaSuccess)
      {
        std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }

      result = cudaDeviceSynchronize();
      if (result != cudaSuccess)
      {
        std::cerr << "cudaDeviceSynchronize() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }

      float elapsed_ms = 0;
      result = cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);
      if (result != cudaSuccess)
      {
        std::cerr << "cudaEventElapsedTime() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }

      for (cudaEvent_t &evt : events)
      {
        result = cudaEventDestroy(evt);
        if (result != cudaSuccess)
        {
          std::cerr << "cudaEventDestroy() failed with error " << cudaGetErrorString(result) << std::endl;
          return false;
        }
      }

      int64_t flops = int64_t(gemm_m) * gemm_n * gemm_k * batch_count * 2;
      int64_t bytes = cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementA>) * int64_t(gemm_m) * int64_t(gemm_k) * batch_count) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementB>) * int64_t(gemm_k) * int64_t(gemm_n) * batch_count) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementD>) * int64_t(gemm_m) * int64_t(gemm_n) * batch_count) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementSFA>) * int64_t(gemm_m) * int64_t(gemm_k) * batch_count / int64_t(kVectorSize)) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementSFB>) * int64_t(gemm_k) * int64_t(gemm_n) * batch_count / int64_t(kVectorSize)) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementSFD>) * int64_t(gemm_m) * int64_t(gemm_n) * batch_count / int64_t(kVectorSize));

      double gflops_per_second = double(flops) * kIterations / double(elapsed_ms / 1000.0f) / double(1.0e9);
      double gbytes_per_second = double(bytes) * kIterations / double(elapsed_ms / 1000.0f) / double(1 << 30);
      double elapsed_ms_per_iter = double(elapsed_ms) / kIterations;

      std::cout << "         Problem: " << gemm_m << "-by-" << gemm_n << "-by-" << gemm_k
                << ", batch size: " << batch_count << std::endl;
      std::cout << "         Runtime: " << elapsed_ms_per_iter << " ms" << std::endl;
      std::cout << "          GFLOPs: " << gflops_per_second << "  GFLOPs" << std::endl;
      std::cout << "Memory bandwidth: " << gbytes_per_second << "  GiB/s" << std::endl;
    }

    if (status != cutlass::Status::kSuccess)
    {
      printf("gemv exec failed\n");
      return false;
    }

    return true;
  }
};

struct Options
{
  bool help = false;
  int m = 4096;
  int k = 2048;
  int batch = 2;
  float alpha = 1.0f;
  float beta = 0.0f;
  float epilogue_st = -1.0f;
  bool profiling = true;
  int iterations = 10;

  void parse(int argc, char const **args)
  {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help"))
    {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("batch", batch);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("epilogue_st", epilogue_st);
    cmd.get_cmd_line_argument("profiling", profiling);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  std::ostream &print_usage(std::ostream &out) const
  {
    out << "batched_fp4_gemv_pytorch\n\n"
        << "  Batched FP4 GEMV with block-scaled inputs and outputs.\n"
        << "  Accepts pre-allocated PyTorch tensors as inputs.\n\n"
        << "Options:\n\n"
        << "  --help                    Display this usage statement\n\n"
        << "  --m=<int>                 M dimension (default: 4096)\n"
        << "  --k=<int>                 K dimension (default: 2048)\n"
        << "  --batch=<int>             Batch count (default: 2)\n"
        << "  --alpha=<f32>             Alpha scalar (default: 1.0)\n"
        << "  --beta=<f32>              Beta scalar (default: 0.0)\n"
        << "  --epilogue_st=<f32>       Epilogue ST value (default: -1.0 for random)\n"
        << "  --profiling=<bool>        Enable profiling (default: true)\n"
        << "  --iterations=<int>        Profiling iterations (default: 10)\n\n"
        << "Examples:\n\n"
        << "$ ./batched_fp4_gemv_pytorch --m=4096 --k=2048 --batch=2\n\n";
    return out;
  }
};

int main(int argc, char const **argv)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  Options options;
  options.parse(argc, argv);

  if (options.help)
  {
    options.print_usage(std::cout);
    return 0;
  }

  using ElementA = cutlass::float_e2m1_t;
  using ElementSFA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;

  using ElementB = cutlass::float_e2m1_t;
  using ElementSFB = cutlass::float_e4m3_t;

  using ElementC = cutlass::float_e2m1_t;

  using ElementD = cutlass::float_e2m1_t;
  using LayoutD = cutlass::layout::ColumnMajor;

  using ElementSFD = cutlass::float_e4m3_t;
  using LayoutSFD = cutlass::layout::ColumnMajor;

  using ElementAccumulatorMainloop = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  static constexpr int kVectorSize = 16;
  static constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ThreadShape = cutlass::gemm::GemmShape<16, 8>;
  static_assert(kVectorSize == ThreadShape::kM, "vector size and thread in row should be equal");

  using EpilogueOp = typename cutlass::epilogue::threadblock::GemvEpilogueWithScalingFactor<kVectorSize,
                                                                                            ThreadShape,
                                                                                            ElementCompute,
                                                                                            ElementAccumulator,
                                                                                            ElementC,
                                                                                            ElementD,
                                                                                            ElementSFD,
                                                                                            LayoutD,
                                                                                            LayoutSFD>;

  using Gemv = cutlass::gemm::device::GemvBlockScaled<
      cutlass::gemm::kernel::
          GemvBlockScaled<ElementA, LayoutA, ElementB, ElementD, ElementAccumulatorMainloop, EpilogueOp, kElementsPerAccess>>;

  const int32_t batch_count = options.batch;
  const int32_t gemm_m = options.m;
  const int32_t gemm_k = options.k;
  const int32_t gemm_n = 1;

  printf("Creating tensors...\n");
  printf("  A: (%d, %d) - batch*M x K\n", batch_count * gemm_m, gemm_k);
  printf("  B: (%d, %d) - batch*K x 1\n", batch_count * gemm_k, 1);
  printf("  D: (%d, %d) - batch*M x 1\n", batch_count * gemm_m, 1);

  torch::Tensor A = torch::randn({batch_count * gemm_m, gemm_k}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
  torch::Tensor B = torch::randn({batch_count * gemm_k, 1}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
  torch::Tensor C = torch::randn({batch_count * gemm_m, 1}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
  torch::Tensor D = torch::zeros({batch_count * gemm_m, 1}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

  using ProblemShapeType = cute::Shape<int, int, int, int>;
  auto problem_shape_MNKL = ProblemShapeType{gemm_m, gemm_n, gemm_k, batch_count};

  using Sm100BlockScaledInputConfig = cutlass::detail::Sm1xxBlockScaledConfig<kVectorSize>;
  using Blk_MN_Input = typename Sm100BlockScaledInputConfig::Blk_MN;
  using Blk_SF_Input = typename Sm100BlockScaledInputConfig::Blk_SF;
  using SfAtom_Input = typename Sm100BlockScaledInputConfig::SfAtom;

  int k_blks_input = cutlass::ceil_div(gemm_k, (int)cute::size<1>(cute::shape(SfAtom_Input{})));
  int m_blks_input = cutlass::ceil_div(gemm_m, (int)cute::size(Blk_MN_Input{}));
  int n_blks_input = cutlass::ceil_div(gemm_n, (int)cute::size(Blk_MN_Input{}));

  int sfa_size = m_blks_input * (int)cute::size(Blk_MN_Input{}) * k_blks_input * (int)cute::size(Blk_SF_Input{}) * batch_count;
  int sfb_size = n_blks_input * (int)cute::size(Blk_MN_Input{}) * k_blks_input * (int)cute::size(Blk_SF_Input{}) * batch_count;

  using Sm1xxBlockScaledOutputConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<kVectorSize, cute::UMMA::Major::MN>;
  auto sfd_layout = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(problem_shape_MNKL);
  int sfd_size = (int)cute::size(cute::filter_zeros(sfd_layout));

  torch::Tensor SFA = torch::randn({sfa_size}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
  torch::Tensor SFB = torch::randn({sfb_size}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
  torch::Tensor SFD = torch::randn({static_cast<int>(sfd_size)}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

  printf("Tensors created successfully.\n\n");

  TestbedPyTorchGemvFp4SFD<Gemv> testbed;

  ElementCompute alpha{options.alpha};
  ElementCompute beta{options.beta};
  const float epilogue_st = options.epilogue_st < 0.f ? static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 5)) : options.epilogue_st;

  printf("Running batched FP4 GEMV kernel...\n");
  printf("  M=%d, K=%d, N=%d, Batch=%d\n", gemm_m, gemm_k, gemm_n, batch_count);
  printf("  alpha=%f, beta=%f, epilogue_st=%f\n", alpha, beta, epilogue_st);
  printf("  Profiling=%s, Iterations=%d\n\n", options.profiling ? "true" : "false", options.iterations);

  bool pass = testbed.run_gemv(A, B, C, D, SFA, SFB, SFD, batch_count, alpha, beta, epilogue_st, options.profiling, options.iterations);

  if (!pass)
  {
    printf("ERROR: Batched FP4 GEMV execution failed\n");
    return 1;
  }

  printf("\nBatched FP4 GEMV execution completed successfully!\n");
  return 0;

#else
  std::cerr << "ERROR: Unsupported GPU architecture. SM100 (Blackwell) required.\n";
  return 1;
#endif
}
