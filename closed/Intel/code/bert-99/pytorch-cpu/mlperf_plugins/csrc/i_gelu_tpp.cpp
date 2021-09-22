#include "i_gelu_tpp.hpp"
#include "el_common_intrin.hpp"

namespace intel_mlperf {

static inline __m512 _mm512_erf_ps(__m512 x) {
  auto a = _mm512_set1_ps(-0.2888f);
  auto b = _mm512_set1_ps(1.0217744f);
  auto c = _mm512_set1_ps(0.0962405432f);

  auto nb = _mm512_set1_ps(1.769f);
  auto m = _mm512_set1_epi32(0x80000000);

  auto ix = _mm512_castps_si512(x);
  auto s = _mm512_and_epi32(m, ix);
  auto abs = _mm512_abs_ps(x);

  auto v = _mm512_min_ps(abs, nb);
  auto y = (a * v + b) * v + c;
  auto z = _mm512_or_epi32(_mm512_castps_si512(y), s);

  return _mm512_castsi512_ps(z);
}

static inline __m512 _mm512_gelu_ps(__m512 x) {
  auto rsqrt_2 = _mm512_set1_ps(0.70710678);
  auto y = _mm512_erf_ps(x * rsqrt_2) + 1;

  return x * 0.5f * y;
}

// Cover only to N [1, 16]
// User's responsibility for tail access
//
template <int vec_l, int N>
struct i32_scale_gelu_scale_i8 {
  inline static void run(
      int8_t *out, int32_t *in, float M,
      float oscale, int8_t o_off);
  inline static void run(
      int8_t *out, int32_t *in, float M,
      float oscale, int8_t o_off, __mmask16 tail);
};

template <int N>
struct i32_scale_gelu_scale_i8<16, N> {
  static constexpr int64_t batch = 16 * N;

  inline static void run(
      int8_t *out, int32_t *in, float M, float oscale, int8_t o_off) {
    auto vM = _mm512_set1_ps(M);
    auto vS = _mm512_set1_ps(oscale);
    auto zoff = _mm_set1_epi8(o_off);

#   pragma unroll N
    for (int i = 0; i < N; ++ i) {
      auto x = _mm512_load_epi32(&in[i * 16]);
      auto f = _mm512_cvtepi32_ps(x) * vM;
      auto o = _mm512_gelu_ps(f);
      auto z = _mm512_scale_minmax_i8_ps(o, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], 0xffff, z, zoff);
    }
  }

  inline static void run(
      int8_t *out, int32_t *in, float M, float oscale, int8_t o_off, __mmask16 tail) {
    auto vM = _mm512_set1_ps(M);
    auto vS = _mm512_set1_ps(oscale);
    auto zoff = _mm_set1_epi8(o_off);

#   pragma unroll N -1
    for (int i = 0; i < N -1; ++ i) {
      auto x = _mm512_load_epi32(&in[i * 16]);
      auto f = _mm512_cvtepi32_ps(x) * vM;
      auto o = _mm512_gelu_ps(f);
      auto z = _mm512_scale_minmax_i8_ps(o, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], 0xffff, z, zoff);
    }
    {
      auto i = N-1;
      auto zero = _mm512_setzero_epi32();
      auto x = _mm512_mask_loadu_epi32(zero, tail, &in[i * 16]);
      auto f = _mm512_cvtepi32_ps(x) * vM;
      auto o = _mm512_gelu_ps(f);
      auto z = _mm512_scale_minmax_i8_ps(o, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], tail, z, zoff);
    }
  }
};

//
// expecting nelem is integer multiply of batch, expand functionality in
// the furture
//
template <int vec_length>
void i_gelu_tpp<vec_length>::ref(
    void *out, void *in, float M, float oscale, int8_t o_off, int64_t nelem) {

  auto constexpr b = i32_scale_gelu_scale_i8<vec_length, 16>::batch;
  auto n_batch = nelem / b;

  auto pin = reinterpret_cast<int32_t *>(in);
  auto pout = reinterpret_cast<int8_t *>(out);

  for (int p = 0; p < n_batch; ++ p, pout += b, pin += b) {
    i32_scale_gelu_scale_i8<vec_length, 16>::run(pout, pin, M, oscale, o_off);
  }
}

template void i_gelu_tpp<16>::ref(void *, void *, float, float, int8_t, int64_t);

template <int vec_l, int N>
struct f32_scale_i8 {
  inline static void run(
      int8_t *out, float *in, float oscale, int8_t o_off);
  inline static void run(
      int8_t *out, float *in, float oscale, int8_t o_off, __mmask16 tail);
};

template <int N>
struct f32_scale_i8<16, N> {
  static constexpr int64_t batch = 16 * N;

  inline static void run(
      int8_t *out, float *in, float oscale, int8_t o_off) {
    auto vS = _mm512_set1_ps(oscale);
    auto zoff = _mm_set1_epi8(o_off);

#   pragma unroll N
    for (int i = 0; i < N; ++ i) {
      auto f = _mm512_load_ps(&in[i * 16]);
      auto z = _mm512_scale_minmax_i8_ps(f, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], 0xffff, z, zoff);
    }
  }

  inline static void run(
      int8_t *out, float *in, float oscale, int8_t o_off, __mmask16 tail) {
    auto vS = _mm512_set1_ps(oscale);
    auto zoff = _mm_set1_epi8(o_off);

#   pragma unroll N -1
    for (int i = 0; i < N -1; ++ i) {
      auto f = _mm512_load_ps(&in[i * 16]);
      auto z = _mm512_scale_minmax_i8_ps(f, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], 0xffff, z, zoff);
    }
    {
      auto i = N-1;
      auto zero = _mm512_setzero_ps();
      auto f = _mm512_mask_loadu_ps(zero, tail, &in[i * 16]);
      auto z = _mm512_scale_minmax_i8_ps(f, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], tail, z, zoff);
    }
  }
};

template <int vec_l, int N>
struct i8_scale_i8 {
  inline static void run(
      int8_t *out, int8_t *in, float oscale, int8_t o_off);
  inline static void run(
      int8_t *out, int8_t *in, float oscale, int8_t o_off, __mmask16 tail);
};

template <int vec_l, int N>
struct i8_scale_f32 {
  inline static void run(
      float *out, int8_t *in, float oscale);
  inline static void run(
      float *out, int8_t *in, float oscale, __mmask16 tail);
};

template <int vec_l, int N>
struct c8_scale_f32 {
  inline static void run(
      float *out, int8_t *in, float oscale);
  inline static void run(
      float *out, int8_t *in, float oscale, __mmask16 tail);
};

template <int N>
struct i8_scale_i8<16, N> {
  static constexpr int64_t batch = 16 * N;

  inline static void run(
      int8_t *out, int8_t *in, float oscale, int8_t o_off) {
    auto vS = _mm512_set1_ps(oscale);
    auto zoff = _mm_set1_epi8(o_off);

#   pragma unroll N
    for (int i = 0; i < N; ++ i) {
      auto f = _mm512_loadu_i8_to_fp32(&in[i * 16]);
      auto z = _mm512_scale_minmax_i8_ps(f, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], 0xffff, z, zoff);
    }
  }

  inline static void run(
      int8_t *out, int8_t *in, float oscale, int8_t o_off, __mmask16 tail) {
    auto vS = _mm512_set1_ps(oscale);
    auto zoff = _mm_set1_epi8(o_off);

#   pragma unroll N -1
    for (int i = 0; i < N -1; ++ i) {
      auto f = _mm512_loadu_i8_to_fp32(&in[i * 16]);
      auto z = _mm512_scale_minmax_i8_ps(f, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], 0xffff, z, zoff);
    }
    {
      auto i = N-1;
      auto zero = _mm512_setzero_ps();
      auto f = _mm512_mask_loadu_ps(zero, tail, &in[i * 16]);
      auto z = _mm512_scale_minmax_i8_ps(f, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], tail, z, zoff);
    }
  }
};

template <int N>
struct i8_scale_f32<16, N> {
  static constexpr int64_t batch = 16 * N;

  inline static void run(
      float *out, int8_t *in, float oscale) {
    auto vS = _mm512_set1_ps(oscale);

#   pragma unroll N
    for (int i = 0; i < N; ++ i) {
      auto f = _mm512_loadu_i8_to_fp32(&in[i * 16]);
      auto z = f * vS;
      _mm512_storeu_ps(&out[i * 16], z);
    }
  }

  inline static void run(
      float *out, int8_t *in, float oscale, __mmask16 tail) {
    auto vS = _mm512_set1_ps(oscale);

#   pragma unroll N -1
    for (int i = 0; i < N -1; ++ i) {
      auto f = _mm512_loadu_i8_to_fp32(&in[i * 16]);
      auto z = f * vS;
      _mm512_storeu_ps(&out[i * 16], z);
    }
    {
    }
  }
};

template <int N>
struct c8_scale_f32<16, N> {
  static constexpr int64_t batch = 16 * N;

  inline static void run(
      float *out, int8_t *in, float oscale) {
    auto vS = _mm512_set1_ps(oscale);

#   pragma unroll N
    for (int i = 0; i < N; ++ i) {
      auto f = _mm512_loadu_c8_to_fp32(&in[i * 16]);
      auto z = f * vS;
      _mm512_storeu_ps(&out[i * 16], z);
    }
  }

  inline static void run(
      float *out, int8_t *in, float oscale, __mmask16 tail) {
    auto vS = _mm512_set1_ps(oscale);

#   pragma unroll N -1
    for (int i = 0; i < N -1; ++ i) {
      auto f = _mm512_loadu_c8_to_fp32(&in[i * 16]);
      auto z = f * vS;
      _mm512_storeu_ps(&out[i * 16], z);
    }
    {
    }
  }
};

//
// expecting nelem is integer multiply of batch, expand functionality in
// the furture
//
template <int vec_length>
void i_identity_tpp<vec_length>::ref(
    int8_t *out, float *in, float oscale, int8_t o_off, int64_t nelem) {

  auto constexpr b = i32_scale_gelu_scale_i8<vec_length, 16>::batch;
  auto n_batch = nelem / b;

  auto pin = in;
  auto pout = reinterpret_cast<int8_t *>(out);

  for (int p = 0; p < n_batch; ++ p, pout += b, pin += b) {
    f32_scale_i8<vec_length, 16>::run(pout, pin, oscale, o_off);
  }
}

template <int vec_length>
void i_identity_tpp<vec_length>::ref(
    int8_t *out, int8_t *in, float oscale, int8_t o_off, int64_t nelem) {

  auto constexpr b = i32_scale_gelu_scale_i8<vec_length, 16>::batch;
  auto n_batch = nelem / b;

  auto pin = in;
  auto pout = reinterpret_cast<int8_t *>(out);

  for (int p = 0; p < n_batch; ++ p, pout += b, pin += b) {
    i8_scale_i8<vec_length, 16>::run(pout, pin, oscale, o_off);
  }
}

template <int vec_length>
void i_identity_tpp<vec_length>::ref_cin(
    float *out, int8_t *in, float oscale, int64_t nelem) {

  auto constexpr b = i32_scale_gelu_scale_i8<vec_length, 16>::batch;
  auto n_batch = nelem / b;

  auto pin = in;
  auto pout = reinterpret_cast<float *>(out);

  for (int p = 0; p < n_batch; ++ p, pout += b, pin += b) {
    c8_scale_f32<vec_length, 16>::run(pout, pin, oscale);
  }
}

template void i_identity_tpp<16>::ref(int8_t *, float *, float, int8_t, int64_t);
template void i_identity_tpp<16>::ref(int8_t *, int8_t *, float, int8_t, int64_t);
template void i_identity_tpp<16>::ref_cin(float *, int8_t *, float, int64_t);

}
