#pragma once
#include <immintrin.h>

namespace intel_mlperf {

template <int vec_length> struct avx_type;

template <> struct avx_type<8> {
  typedef __m256 type;
};

template <> struct avx_type<16> {
  typedef __m512 type;
};

template <int vec_length>
class i_softmax_tpp {
public:
  i_softmax_tpp(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) :
    dim0(dim0), dim1(dim1), dim2(dim2), dim3(dim3) {
  }
  void ref(void *out, void *in, float *att_mask, float M, float oscale);
  void ref(void *out, void *in, int32_t* att_lens, float M, float oscale);

protected:
  using __mtype = typename avx_type<vec_length>::type;

  int64_t dim0, dim1, dim2, dim3;
};

}
