#include "test/test.h"
#include "aligned.h"
#include "kernels.h"

#include <numeric>

namespace intgemm {

template <CPUType CPUType_>
void kernel_quantize_test() {
  if (kCPU < CPUType_)
    return;

  using input_vec_t = vector_t<CPUType_, float>;
  using output_vec_t = vector_t<CPUType_, int>;

  AlignedVector<float> input(sizeof(input_vec_t) / sizeof(float));
  AlignedVector<int> output(sizeof(output_vec_t) / sizeof(int));

  std::iota(input.begin(), input.end(), 0);
  auto quant_mult = set1_ps<input_vec_t>(2.f);

  *output.template as<output_vec_t>() = kernels::quantize(*input.template as<input_vec_t>(), quant_mult);
  for (auto i = 0; i < output.size(); ++i)
    CHECK(output[i] == int(i*2.f));
}

template INTGEMM_SSE2 void kernel_quantize_test<CPUType::SSE2>();
TEST_CASE("Kernel: quantize SSE2",) { return kernel_quantize_test<CPUType::SSE2>(); }

template INTGEMM_AVX2 void kernel_quantize_test<CPUType::AVX2>();
TEST_CASE("Kernel: quantize AVX2",) { return kernel_quantize_test<CPUType::AVX2>(); }

#ifndef INTGEMM_NO_AVX512
template INTGEMM_AVX512BW void kernel_quantize_test<CPUType::AVX512BW>();
TEST_CASE("Kernel: quantize AVX512BW",) { return kernel_quantize_test<CPUType::AVX512BW>(); }
#endif

}
