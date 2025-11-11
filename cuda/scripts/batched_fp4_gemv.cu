#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <optional>

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
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

template <typename Element, typename Layout>
bool initialize_tensor(cutlass::TensorView<Element, Layout> view, cutlass::Distribution::Kind dist_kind, uint64_t seed) {
  if (dist_kind == cutlass::Distribution::Uniform) {
    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    } else if (bits_input <= 6) {
      scope_max = 2;
      scope_min = -2;
    } else if (bits_input <= 8) {
      if constexpr (cutlass::is_same_v<Element, cutlass::float_ue4m3_t> ||
                    cutlass::is_same_v<Element, cutlass::float_ue8m0_t>) {
        scope_max = 4;
        scope_min = 1;
      } else {
        scope_max = 1;
        scope_min = -1;
      }
    } else {
      scope_max = 4;
      scope_min = -4;
    }

    cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
  } else if (dist_kind == cutlass::Distribution::Identity) {
    cutlass::reference::host::TensorFillIdentity(view);
  } else if (dist_kind == cutlass::Distribution::Gaussian) {
    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  } else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  } else if (dist_kind == cutlass::Distribution::AllOnes) {
    cutlass::reference::host::TensorFill(view, Element(1));
  } else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view, Element(0));
  } else {
    CUTLASS_ASSERT(false);
    return false;
  }

  return true;
}

template <
  typename Gemv_,
  typename ElementC, typename LayoutC, typename ElementD_,
  typename LayoutD, typename ElementSFD_, typename LayoutSFD,
  typename ElementCompute_, int kVectorSize_>
struct TestbedBatchedGemvFp4SFDBase {
public:
  using Gemv = Gemv_;

  using ElementA = typename Gemv::ElementA;
  using ElementSFA = typename Gemv::ElementSFA;
  using LayoutA = typename Gemv::LayoutA;
  static_assert(cutlass::is_same_v<LayoutA, cutlass::layout::RowMajor>, "only support row major matrix A");
  static_assert(cutlass::sizeof_bits<ElementSFA>::value == 8, "ElementSFA should be FP8 type");

  using ElementB = typename Gemv::ElementB;
  using ElementSFB = typename Gemv::ElementSFB;
  using LayoutB = cutlass::layout::ColumnMajor;
  static_assert(cutlass::is_same_v<ElementA, ElementB>, "only support ElementA ElementB of same type");
  static_assert(cutlass::sizeof_bits<ElementSFB>::value == 8, "ElementSFB should be FP8 type");

  static_assert(cutlass::is_same_v<LayoutC, cutlass::layout::ColumnMajor>, "only support col major output D");

  using ElementD = ElementD_;
  static_assert(cutlass::is_same_v<LayoutD, cutlass::layout::ColumnMajor>, "only support col major output D");

  using ElementSFD = ElementSFD_;
  static_assert(cutlass::is_same_v<LayoutSFD, cutlass::layout::ColumnMajor>, "only support col major output SFD");
  static_assert(cutlass::sizeof_bits<ElementSFD>::value, "only support 8 bit SFD");

  using ElementAccumulator = typename Gemv::ElementAccumulator;
  using ElementCompute = ElementCompute_;
  static_assert(cutlass::is_same_v<ElementCompute, float>, "only support fp32 epi compute");

  static constexpr int kVectorSize = kVectorSize_;
  static_assert(kVectorSize == 16, "only support vs 16");

  static constexpr bool kIsKMajorSFD = cutlass::is_same_v<LayoutSFD, cutlass::layout::RowMajor>;
  using Sm1xxBlockScaledOutputConfig =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<kVectorSize,
                                                    kIsKMajorSFD ? cute::UMMA::Major::K : cute::UMMA::Major::MN>;
  using Blk_MN_Output = typename Sm1xxBlockScaledOutputConfig::Blk_MN;
  using Blk_SF_Output = typename Sm1xxBlockScaledOutputConfig::Blk_SF;
  using OutputSFAtom = typename Sm1xxBlockScaledOutputConfig::SfAtom;

  using Sm100BlockScaledInputConfig = cutlass::detail::Sm1xxBlockScaledConfig<kVectorSize>;
  using Blk_MN_Input = typename Sm100BlockScaledInputConfig::Blk_MN;
  using Blk_SF_Input = typename Sm100BlockScaledInputConfig::Blk_SF;
  using SfAtom_Input = typename Sm100BlockScaledInputConfig::SfAtom;

public:
  TestbedBatchedGemvFp4SFDBase(cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
                    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
                    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
                    cutlass::Distribution::Kind init_D_ = cutlass::Distribution::Uniform,
                    cutlass::Distribution::Kind init_SFA_ = cutlass::Distribution::Uniform,
                    cutlass::Distribution::Kind init_SFB_ = cutlass::Distribution::Uniform,
                    cutlass::Distribution::Kind init_SFD_ = cutlass::Distribution::Uniform,
                    uint64_t seed_ = 2023)
      : init_A(init_A_), init_B(init_B_), init_C(init_C_), init_D(init_D_),
        init_SFA(init_SFA_), init_SFB(init_SFB_), init_SFD(init_SFD_), seed(seed_) {}

  bool initialize(cutlass::MatrixCoord problem_size, int32_t batch_count) {
    const int32_t gemm_m = problem_size.row();
    const int32_t gemm_k = problem_size.column();
    const int32_t gemm_n = 1;
    const int32_t gemm_batch = batch_count;

    auto k_blks_input = cutlass::ceil_div(gemm_k, cute::size<1>(shape(SfAtom_Input{})));
    auto m_blks_input = cutlass::ceil_div(gemm_m, Blk_MN_Input{});
    auto n_blks_input = cutlass::ceil_div(gemm_n, Blk_MN_Input{});

    auto sfa_coord = cutlass::make_Coord(m_blks_input * Blk_MN_Input{} * gemm_batch, k_blks_input * Blk_SF_Input{});
    auto sfb_coord = cutlass::make_Coord(n_blks_input * Blk_MN_Input{} * gemm_batch, k_blks_input * Blk_SF_Input{});

    auto sfa_resize_layout =
        cutlass::layout::Affine2Layout_Factory<LayoutA>::layout_factory(sfa_coord, typename LayoutA::Stride{});
    auto sfb_resize_layout =
        cutlass::layout::Affine2Layout_Factory<LayoutB>::layout_factory(sfb_coord, typename LayoutB::Stride{});

    using ProblemShapeType = cute::Shape<int, int, int, int>;
    auto problem_shape_MNKL = ProblemShapeType{gemm_m, gemm_n, gemm_k, gemm_batch};
    auto sfd_layout = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(problem_shape_MNKL);
    auto sfd_size = cute::size(cute::filter_zeros(sfd_layout));
    auto sfd_coord = cutlass::make_Coord(sfd_size, 1);

    auto sfd_resize_layout =
        cutlass::layout::Affine2Layout_Factory<LayoutSFD>::layout_factory(sfd_coord, typename LayoutSFD::Stride{});

    this->reference_D.resize({gemm_batch * gemm_m, 1});
    this->reference_SFD.resize(sfd_coord, sfd_resize_layout);

    if (initialize_tensor(this->reference_D.host_view(), this->init_D, this->seed + 7) == false) {
      printf("initialize_tensor() REF D failed\n");
      return false;
    }
    if (initialize_tensor(this->reference_SFD.host_view(), this->init_SFD, this->seed + 9) == false) {
      printf("initialize_tensor() REF SFD failed\n");
      return false;
    }

    this->tensor_A.resize({gemm_batch * gemm_m, gemm_k});
    this->tensor_B.resize({gemm_batch * gemm_k, 1});
    this->tensor_C.resize({gemm_batch * gemm_m, 1});
    this->tensor_D.resize({gemm_batch * gemm_m, 1});
    this->tensor_SFA.resize(sfa_coord, sfa_resize_layout);
    this->tensor_SFB.resize(sfb_coord, sfb_resize_layout);
    this->tensor_SFD.resize(sfd_coord, sfd_resize_layout);

    if (initialize_tensor(this->tensor_A.host_view(), this->init_A, this->seed + 1) == false) {
      printf("initialize_tensor() A failed\n");
      return false;
    }
    if (initialize_tensor(this->tensor_B.host_view(), this->init_B, this->seed + 2) == false) {
      printf("initialize_tensor() B failed\n");
      return false;
    }
    if (initialize_tensor(this->tensor_C.host_view(), this->init_C, this->seed + 3) == false) {
      printf("initialize_tensor() C failed\n");
      return false;
    }

    if (initialize_tensor(this->tensor_SFA.host_view(), this->init_SFA, this->seed + 4) == false) {
      printf("initialize_tensor() SFA failed\n");
      return false;
    }
    if (initialize_tensor(this->tensor_SFB.host_view(), this->init_SFB, this->seed + 5) == false) {
      printf("initialize_tensor() SFB failed\n");
      return false;
    }

    if (initialize_tensor(this->tensor_D.host_view(), this->init_D, this->seed + 6) == false) {
      printf("initialize_tensor() D failed\n");
      return false;
    }
    if (initialize_tensor(this->tensor_SFD.host_view(), this->init_SFD, this->seed + 8) == false) {
      printf("initialize_tensor() SFD failed\n");
      return false;
    }

    this->tensor_A.sync_device();
    this->tensor_B.sync_device();
    this->tensor_C.sync_device();
    this->tensor_D.sync_device();
    this->tensor_SFA.sync_device();
    this->tensor_SFB.sync_device();
    this->tensor_SFD.sync_device();

    cutlass::device_memory::copy_to_host(this->reference_SFD.host_data(), this->tensor_SFD.device_data(), sfd_size);

    return true;
  }

  bool compare_reference() {
    this->tensor_D.sync_host();

    bool passed = true;

    passed = cutlass::reference::host::TensorEquals(this->reference_D.host_view(), this->tensor_D.host_view());
    if (passed == false) {
      printf("gemm_m: %d, gemm_k: %d, ", this->tensor_A.host_view().extent(0), this->tensor_A.host_view().extent(1));
      printf("tensorD mismatch\n");
      return false;
    }

    this->tensor_SFD.sync_host();

    passed = cutlass::reference::host::TensorEquals(this->reference_SFD.host_view(), this->tensor_SFD.host_view());
    if (passed == false) {
      printf("gemm_m: %d, gemm_k: %d, ", this->tensor_A.host_view().extent(0), this->tensor_A.host_view().extent(1));
      printf("tensorSFD mismatch\n");
      return false;
    }

    return passed;
  }

  bool run_reference(cutlass::MatrixCoord problem_size,
                     int32_t batch_count,
                     ElementCompute alpha,
                     ElementCompute beta,
                     float epilogue_st) {
    const int32_t gemm_m = problem_size.row();
    const int32_t gemm_k = problem_size.column();
    const int32_t gemm_n = 1;
    const int32_t gemm_batch = batch_count;

    using ProblemShapeType = cute::Shape<int, int, int, int>;
    auto problem_shape_MNKL = ProblemShapeType{gemm_m, gemm_n, gemm_k, gemm_batch};
    auto SfD = make_tensor(make_iterator(this->reference_SFD.host_data()),
                           Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(problem_shape_MNKL));

    using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
    using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
    using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(gemm_m, gemm_k, gemm_batch));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(gemm_n, gemm_k, gemm_batch));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(gemm_m, gemm_n, gemm_batch));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(gemm_m, gemm_n, gemm_batch));

    auto A = make_tensor(make_iterator(this->tensor_A.host_data()),
                         cute::make_layout(cute::make_shape(gemm_m, gemm_k, gemm_batch), stride_a));
    auto B = make_tensor(make_iterator(this->tensor_B.host_data()),
                         cute::make_layout(cute::make_shape(gemm_n, gemm_k, gemm_batch), stride_b));
    auto C = cute::make_tensor(make_iterator(this->tensor_C.host_data()),
                               cute::make_layout(cute::make_shape(gemm_m, gemm_n, gemm_batch), stride_c));
    auto D = cute::make_tensor(make_iterator(this->reference_D.host_data()),
                               cute::make_layout(cute::make_shape(gemm_m, gemm_n, gemm_batch), stride_d));

    auto layout_sfa = Sm100BlockScaledInputConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
    auto layout_sfb = Sm100BlockScaledInputConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

    auto SfA = make_tensor(this->tensor_SFA.host_data(), layout_sfa);
    auto SfB = make_tensor(this->tensor_SFB.host_data(), layout_sfb);

    typename cutlass::reference::host::GettBlockScalingMainloopParams<ElementAccumulator,
                                                                      decltype(A),
                                                                      decltype(SfA),
                                                                      decltype(B),
                                                                      decltype(SfB)>
        mainloop_params{A, SfA, B, SfB};

    typename cutlass::reference::host::GettBlockScalingEpilogueParams<ElementCompute,
                                                                      ElementAccumulator,
                                                                      ElementCompute,
                                                                      decltype(C),
                                                                      decltype(D),
                                                                      decltype(SfD),
                                                                      cute::Int<kVectorSize>,
                                                                      cutlass::reference::host::SfStrategy::SfDGen>
        epilogue_params{alpha, beta, C, D, SfD, epilogue_st};

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    return true;
  }

  virtual typename Gemv::Arguments get_arguments(
    cutlass::MatrixCoord problem_size, int32_t batch_count,
    float epilogue_st, ElementCompute alpha, ElementCompute beta) = 0;

  bool run_gemv(cutlass::MatrixCoord problem_size,
                int32_t batch_count,
                ElementCompute alpha,
                ElementCompute beta,
                [[maybe_unused]] float epilogue_st,
                bool is_profiling,
                int kIterations) {

    const int32_t gemm_m = problem_size.row();
    const int32_t gemm_k = problem_size.column();
    [[maybe_unused]] const int32_t gemm_n = 1;
    [[maybe_unused]] const int32_t gemm_batch = batch_count;

    Gemv gemv_op;
    typename Gemv::Arguments arguments = this->get_arguments(
      problem_size, batch_count, epilogue_st, alpha, beta
    );

    cutlass::Status status = gemv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      printf("can_implement() failed\n");
      return false;
    }

    size_t workspace_size = Gemv::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemv_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      printf("initialize() failed\n");
      return false;
    }

    if (not is_profiling) {
      status = gemv_op();
    } else {
      cudaError_t result;
      cudaEvent_t events[2];

      for (cudaEvent_t &evt : events) {
        result = cudaEventCreate(&evt);
        if (result != cudaSuccess) {
          std::cerr << "cudaEventCreate failed with error " << cudaGetErrorString(result) << std::endl;
          return false;
        }
      }

      status = gemv_op();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Device execution failed on warmup." << std::endl;
        return false;
      }

      result = cudaEventRecord(events[0]);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }

      for (int iter_i = 0; iter_i < kIterations; ++iter_i) {
        status = gemv_op();
        if (status != cutlass::Status::kSuccess) {
          std::cerr << "Device execution failed." << std::endl;
          return false;
        }
      }

      result = cudaEventRecord(events[1]);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }

      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }

      float elapsed_ms = 0;
      result = cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventElapsedTime() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }

      for (cudaEvent_t &evt : events) {
        result = cudaEventDestroy(evt);
        if (result != cudaSuccess) {
          std::cerr << "cudaEventDestroy() failed with error " << cudaGetErrorString(result) << std::endl;
          return false;
        }
      }

      int64_t flops = int64_t(gemm_m) * gemm_n * gemm_k * gemm_batch * 2;
      int64_t bytes = cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementA>) * int64_t(gemm_m) * int64_t(gemm_k) * gemm_batch) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementB>) * int64_t(gemm_k) * int64_t(gemm_n) * gemm_batch) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementD>) * int64_t(gemm_m) * int64_t(gemm_n) * gemm_batch) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementSFA>) * int64_t(gemm_m) * int64_t(gemm_k) * gemm_batch / int64_t(kVectorSize)) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementSFB>) * int64_t(gemm_k) * int64_t(gemm_n) * gemm_batch / int64_t(kVectorSize)) +
                      cutlass::bits_to_bytes<int64_t>(int64_t(cute::sizeof_bits_v<ElementSFD>) * int64_t(gemm_m) * int64_t(gemm_n) * gemm_batch / int64_t(kVectorSize));

      double gflops_per_second = double(flops) * kIterations / double(elapsed_ms / 1000.0f) / double(1.0e9);
      double gbytes_per_second = double(bytes) * kIterations / double(elapsed_ms / 1000.0f) / double(1 << 30);
      double elapsed_ms_per_iter = double(elapsed_ms) / kIterations;

      std::cout << "         Problem: "
                << gemm_m << "-by-" << gemm_n << "-by-" << gemm_k
                << ", batch size: " << gemm_batch
                << std::endl;
      std::cout << "         Runtime: " << elapsed_ms_per_iter << " ms" << std::endl;
      std::cout << "          GFLOPs: " << gflops_per_second << "  GFLOPs" << std::endl;
      std::cout << "Memory bandwidth: " << gbytes_per_second << "  GiB/s" << std::endl;
    }

    if (status != cutlass::Status::kSuccess) {
      printf("gemv exec failed\n");
      return false;
    }

    return true;
  }

  bool run_and_verify(cutlass::MatrixCoord problem_size,
           int32_t batch_count,
           ElementCompute alpha,
           ElementCompute beta,
           float epilogue_st) {

    if (this->initialize(problem_size, batch_count) == false) {
      return false;
    }

    if (this->run_gemv(problem_size, batch_count, alpha, beta, epilogue_st, false, 1) == false) {
      return false;
    }

    if (this->run_reference(problem_size, batch_count, alpha, beta, epilogue_st) == false) {
      printf("run_reference() failed\n");
      return false;
    }

    if (this->compare_reference() == false) {
      printf("compare_reference() failed\n");
      return false;
    }

    return true;
  }

  bool profile(cutlass::MatrixCoord problem_size,
           int32_t batch_count,
           ElementCompute alpha,
           ElementCompute beta,
           float epilogue_st,
           int kIterations = 10) {

    if (this->initialize(problem_size, batch_count) == false) {
      return false;
    }

    if (this->run_gemv(problem_size, batch_count, alpha, beta, epilogue_st, true, kIterations) == false) {
      return false;
    }

    return true;
  }

public:
  cutlass::HostTensor<ElementA, LayoutA> tensor_A;
  cutlass::HostTensor<ElementSFA, LayoutA> tensor_SFA;

  cutlass::HostTensor<ElementB, LayoutB> tensor_B;
  cutlass::HostTensor<ElementSFB, LayoutB> tensor_SFB;

  cutlass::HostTensor<ElementC, LayoutC> tensor_C;

  cutlass::HostTensor<ElementD, LayoutD> tensor_D;
  cutlass::HostTensor<ElementSFD, LayoutD> tensor_SFD;

  cutlass::HostTensor<ElementD, LayoutD> reference_D;
  cutlass::HostTensor<ElementSFD, LayoutD> reference_SFD;

  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_D;
  cutlass::Distribution::Kind init_SFA;
  cutlass::Distribution::Kind init_SFB;
  cutlass::Distribution::Kind init_SFD;
  uint64_t seed;
};

template<typename Gemv_>
struct TestbedBatchedGemvFp4SFD : public TestbedBatchedGemvFp4SFDBase<
  Gemv_,
  typename Gemv_::ElementC,
  typename Gemv_::EpilogueOutputOp::LayoutOutput,
  typename Gemv_::EpilogueOutputOp::ElementD,
  typename Gemv_::EpilogueOutputOp::LayoutOutput,
  typename Gemv_::EpilogueOutputOp::ElementSFD,
  typename Gemv_::EpilogueOutputOp::LayoutSFD,
  typename Gemv_::EpilogueOutputOp::ElementCompute,
  Gemv_::EpilogueOutputOp::kVectorSize
> {
  using Base = TestbedBatchedGemvFp4SFDBase<
    Gemv_,
    typename Gemv_::ElementC,
    typename Gemv_::EpilogueOutputOp::LayoutOutput,
    typename Gemv_::EpilogueOutputOp::ElementD,
    typename Gemv_::EpilogueOutputOp::LayoutOutput,
    typename Gemv_::EpilogueOutputOp::ElementSFD,
    typename Gemv_::EpilogueOutputOp::LayoutSFD,
    typename Gemv_::EpilogueOutputOp::ElementCompute,
    Gemv_::EpilogueOutputOp::kVectorSize
  >;

  using Base::Base;
  using Gemv = Gemv_;
  using ElementCompute = typename Base::ElementCompute;
  using SfAtom_Input = typename Base::SfAtom_Input;
  using Blk_MN_Input = typename Base::Blk_MN_Input;
  using Blk_SF_Input = typename Base::Blk_SF_Input;

  static constexpr int kVectorSize = Base::kVectorSize;

  typename Gemv::Arguments get_arguments(
    cutlass::MatrixCoord problem_size,
    int32_t batch_count, float epilogue_st,
    ElementCompute alpha, ElementCompute beta) override {

    const int32_t gemm_m = problem_size.row();
    const int32_t gemm_k = problem_size.column();
    [[maybe_unused]] const int32_t gemm_n = 1;
    [[maybe_unused]] const int32_t gemm_batch = batch_count;

    auto k_blks_input = cutlass::ceil_div(gemm_k, cute::size<1>(shape(SfAtom_Input{})));
    auto m_blks_input = cutlass::ceil_div(gemm_m, Blk_MN_Input{});
    auto n_blks_input = cutlass::ceil_div(gemm_n, Blk_MN_Input{});

    int batch_stride_SFA = m_blks_input * Blk_MN_Input{} * k_blks_input * Blk_SF_Input{};
    int batch_stride_SFB = n_blks_input * Blk_MN_Input{} * k_blks_input * Blk_SF_Input{};

    using ProblemShapeType = cute::Shape<int, int, int, int>;
    auto problem_shape_MNKL = ProblemShapeType{gemm_m, gemm_n, gemm_k, gemm_batch};

    using Sm1xxBlockScaledOutputConfig = typename Base::Sm1xxBlockScaledOutputConfig;
    auto sfd_layout = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(problem_shape_MNKL);

    auto batch_stride_tuple = cute::stride<2>(sfd_layout);
    int batch_stride_SFD = static_cast<int>(cute::get<1>(batch_stride_tuple));

    typename Gemv::Arguments arguments{
        problem_size,
        batch_count,
        typename Gemv::EpilogueOutputOp::Params{
            this->tensor_D.device_ref(),
            this->tensor_SFD.device_data(),
            alpha,
            beta,
            epilogue_st,
            batch_stride_SFD,
            gemm_m
        },
        this->tensor_A.device_ref(),
        this->tensor_B.device_data(),
        this->tensor_C.device_data(),
        this->tensor_D.device_data(),
        this->tensor_SFA.device_data(),
        this->tensor_SFB.device_data(),
        gemm_k,
        gemm_m * gemm_k,
        gemm_k,
        gemm_m,
        gemm_m,
        batch_stride_SFA,
        batch_stride_SFB,
        batch_stride_SFD
    };

    return arguments;
  }
};

struct Options {
  bool help = false;

  int m = 4096;
  int k = 2048;
  int n = 1;
  int batch = 2;

  float alpha = 1.0f;
  float beta = 0.0f;
  float epilogue_st = -1.0f;

  bool profiling = true;
  int iterations = 10;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("batch", batch);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta",  beta);
    cmd.get_cmd_line_argument("epilogue_st",  epilogue_st);
    cmd.get_cmd_line_argument("profiling", profiling);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  std::ostream & print_usage(std::ostream &out) const {

    out << "batched_fp4_gemv\n\n"
      << "  Batched FP4 GEMV with block-scaled inputs and outputs.\n\n"
      << "Options:\n\n"
      << "  --help                                                       If specified, displays this usage statement\n\n"
      << "  --m=<int>                                                    Sets the M extent of the GEMM\n"
      << "  --k=<int>                                                    Sets the K extent of the GEMM\n"
      << "  --batch=<int>                                                Sets the batch count of the GEMM\n"
      << "  --alpha=<f32>                                                Epilogue scalar alpha\n"
      << "  --beta=<f32>                                                 Epilogue scalar beta\n"
      << "  --epilogue_st=<f32>                                          Epilogue ST value\n\n"
      << "  --profiling=<bool>                                           Whether to run profiling\n\n"
      << "  --iterations=<int>                                           Number of profiling iterations to perform\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "batched_fp4_gemv" << " --m=4096 --k=2048 --batch=2 \n\n";

    return out;
  }
};

bool run_batched_fp4_gemv_device(Options const& options) {
  CUTLASS_ASSERT(options.n == 1);

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

  ElementCompute alpha{options.alpha};
  ElementCompute beta{options.beta};
  const float epilogue_st = options.epilogue_st < 0.f ?
    static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 5)) :
    options.epilogue_st;

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

  TestbedBatchedGemvFp4SFD<Gemv> testbed;

  bool pass = true;

  if (options.profiling) {
    pass = testbed.profile(cutlass::MatrixCoord{options.m, options.k}, options.batch, alpha, beta, epilogue_st, options.iterations);
  } else {
    pass = testbed.run_and_verify(cutlass::MatrixCoord{options.m, options.k}, options.batch, alpha, beta, epilogue_st);
  }

  return pass;
}

int main(int argc, char const** argv) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  Options verification_options = options;
  verification_options.profiling = false;

  bool passed = run_batched_fp4_gemv_device(verification_options);
  if (passed == false) {
    printf("test fail\n");
    return 1;
  } else {
    printf("test pass\n");
  }

  if (options.profiling) {
    printf("\nProfiling...\n");
    passed = run_batched_fp4_gemv_device(options);
    if (passed == false) {
      printf("profiling fail\n");
      return 1;
    } else {
      printf("profiling completed\n");
    }
  }

  return 0;
#else
  std::cerr << "Unsupported example. Please ensure CUTLASS_ARCH_MMA_SM100_SUPPORTED is defined.\n";
  return 0;
#endif
}
