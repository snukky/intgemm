#include "callbacks/configs.h"

#include "intrinsics.h"
#include "kernels.h"
#include "types.h"
#include "vec_traits.h"

#if defined(THIS_IS_SSE2)
  #define CPU_NAME SSE2
  #define CPU_ATTR INTGEMM_SSE2
#elif defined(THIS_IS_AVX2)
  #define CPU_NAME AVX2
  #define CPU_ATTR INTGEMM_AVX2
#elif defined(THIS_IS_AVX512BW)
  #define CPU_NAME AVX512BW
  #define CPU_ATTR INTGEMM_AVX512BW
#else
  #error "Only SSE2, AVX2 and AVX512BW are supported"
#endif

#define vi vector_t<CPUType::CPU_NAME, int>
#define vf vector_t<CPUType::CPU_NAME, float>
#define vd vector_t<CPUType::CPU_NAME, double>
#define dvi dvector_t<CPUType::CPU_NAME, int>
#define dvf dvector_t<CPUType::CPU_NAME, float>
#define dvd dvector_t<CPUType::CPU_NAME, double>

#if defined(THIS_IS_SSE2)
  #define vinput dvector_t<CPUType::SSE2, int>
  #define vinput_i dvector_t<CPUType::SSE2, int>
  #define vinput_f dvector_t<CPUType::SSE2, float>
  #define vinput_d dvector_t<CPUType::SSE2, double>
  #define vinput_vi vector_t<CPUType::SSE2, int>
  #define vinput_vf vector_t<CPUType::SSE2, float>
  #define vinput_vd vector_t<CPUType::SSE2, double>
#else
  #define vinput vector_t<CPUType::AVX2, int>
  #define vinput_i vector_t<CPUType::AVX2, int>
  #define vinput_f vector_t<CPUType::AVX2, float>
  #define vinput_d vector_t<CPUType::AVX2, double>
  #define vinput_vi vector_t<CPUType::AVX2, int>
  #define vinput_vf vector_t<CPUType::AVX2, float>
  #define vinput_vd vector_t<CPUType::AVX2, double>
#endif

namespace intgemm {
namespace callbacks {

template <CPUType CpuType, typename... CallbackConfigs>
class CallbackImpl;

}}

/*
 * Callbacks implementations....
 */
namespace intgemm {
namespace callbacks {

template <typename... CallbackConfigs>
class CallbackImpl<CPUType::CPU_NAME, CallbackConfigs...> {
public:
  CPU_ATTR CallbackImpl(const CallbackConfigs&... configs)
    : callbacks(std::make_tuple(CallbackImpl<CPUType::CPU_NAME, CallbackConfigs>(configs)...)) {}

  CPU_ATTR void operator()(vinput input, Index A_rowidx, Index B_colidx, Index A_rows, Index width, Index B_cols) {
    run_callbacks_pipeline(input, A_rowidx, B_colidx, A_rows, width, B_cols, callbacks, make_sequence<sizeof...(CallbackConfigs)>());
  }

private:
  const std::tuple<CallbackImpl<CPUType::CPU_NAME, CallbackConfigs>...> callbacks;

#define RUN_CALLBACKS_PIPELINE_IMPL(vtype) \
  template <typename Tuple, unsigned FirstIndex> \
  CPU_ATTR static inline void run_callbacks_pipeline(vtype input, Index A_rowidx, Index B_colidx, Index A_rows, Index width, Index B_cols, Tuple tuple, sequence<FirstIndex>) { \
    std::get<FirstIndex>(tuple)(input, A_rowidx, B_colidx, A_rows, width, B_cols); \
  } \
  template <typename Tuple, unsigned FirstIndex, unsigned SecondIndex, unsigned... RestIndices> \
  CPU_ATTR static inline void run_callbacks_pipeline(vtype input, Index A_rowidx, Index B_colidx, Index A_rows, Index width, Index B_cols, Tuple tuple, sequence<FirstIndex, SecondIndex, RestIndices...>) { \
    auto output = std::get<FirstIndex>(tuple)(input, A_rowidx, B_colidx, A_rows, width, B_cols); \
    run_callbacks_pipeline(output, A_rowidx, B_colidx, A_rows, width, B_cols, tuple, sequence<SecondIndex, RestIndices...>()); \
  }

  RUN_CALLBACKS_PIPELINE_IMPL(vinput_i)
  RUN_CALLBACKS_PIPELINE_IMPL(vinput_f)
  RUN_CALLBACKS_PIPELINE_IMPL(vinput_d)

#undef RUN_CALLBACKS_PIPELINE_IMPL
};


/*
 * Dummy
 */
template <> class CallbackImpl<CPUType::CPU_NAME, Dummy> {
public:
  CPU_ATTR CallbackImpl(const Dummy&) {}
  CPU_ATTR void operator()(vinput, Index, Index, Index, Index, Index) {}
};

/*
 * Unquantize
 */
template <> class CallbackImpl<CPUType::CPU_NAME, Unquantize> {
public:
  CPU_ATTR CallbackImpl(const Unquantize& config) {
    unquant_mult = set1_ps<vinput_vf>(config.unquant_mult);
  }

  CPU_ATTR vinput_f operator()(vinput_i input, Index A_rowidx, Index B_colidx, Index A_rows, Index width, Index B_cols) {
    return kernels::unquantize(input, unquant_mult);
  }
private:
  vinput_vf unquant_mult;
};

/*
 * Write
 */
template <> class CallbackImpl<CPUType::CPU_NAME, Write> {
public:
  CPU_ATTR CallbackImpl(const Write& config) : config(config) {}

  CPU_ATTR void operator()(vinput_f input, Index A_rowidx, Index B_colidx, Index A_rows, Index width, Index B_cols) {
    kernels::write(input, config.addr, A_rowidx * B_cols + B_colidx);
  }
private:
  Write config;
};

/*
 * AddBias
 */
template <> class CallbackImpl<CPUType::CPU_NAME, AddBias> {
public:
  CPU_ATTR CallbackImpl(const AddBias& config) : config(config) {}

  CPU_ATTR vinput_f operator()(vinput_f input, Index A_rowidx, Index B_colidx, Index A_rows, Index width, Index B_cols) {
    return kernels::add_bias(input, config.bias_addr, B_colidx);
  }
private:
  AddBias config;
};

/*
 * UnquantizeAndWrite
 */
template <> class CallbackImpl<CPUType::CPU_NAME, UnquantizeAndWrite> {
public:
  CPU_ATTR CallbackImpl(const UnquantizeAndWrite& config) : config(config) {
    unquant_mult = set1_ps<vinput_vf>(config.unquant_mult);
  }

  CPU_ATTR void operator()(vinput input, Index A_rowidx, Index B_colidx, Index A_rows, Index width, Index B_cols) {
    auto result = kernels::unquantize(input, unquant_mult);
    kernels::write(result, config.addr, A_rowidx * B_cols + B_colidx);
  }

private:
  UnquantizeAndWrite config;
  vinput_vf unquant_mult;
};

/*
 * UnquantizeAndAddBiasAndWrite
 */
template <> class CallbackImpl<CPUType::CPU_NAME, UnquantizeAndAddBiasAndWrite> {
public:
  CPU_ATTR CallbackImpl(const UnquantizeAndAddBiasAndWrite& config) : config(config) {
    unquant_mult = set1_ps<vinput_vf>(config.unquant_mult);
  }

  CPU_ATTR void operator()(vinput input, Index A_rowidx, Index B_colidx, Index A_rows, Index width, Index B_cols) {
    auto result = kernels::unquantize(input, unquant_mult);
    result = kernels::add_bias(result, config.bias_addr, B_colidx);
    kernels::write(result, config.output_addr, A_rowidx * B_cols + B_colidx);
  }

private:
  UnquantizeAndAddBiasAndWrite config;
  vinput_vf unquant_mult;
};

}
}

#undef CPU_NAME
#undef CPU_ATTR
#undef vi
#undef vf
#undef vd
#undef dvi
#undef dvf
#undef dvd
#undef vinput
#undef vinput_i
#undef vinput_f
#undef vinput_d
#undef vinput_vi
#undef vinput_vf
#undef vinput_vd
