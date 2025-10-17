
#include <hwy/highway.h>

#include <complex>
#include <cstdint>
#include <type_traits>

#include "../CPPMinimalFFT.hpp"
#include "../plan.hpp"
#include "../weights.hpp"

namespace hn = hwy::HWY_NAMESPACE;

#define CCDPTR(x) reinterpret_cast<const double *__restrict__>(__builtin_assume_aligned(x, 16))
#define CDPTR(x) reinterpret_cast<double *__restrict__>(__builtin_assume_aligned(x, 16))
#define CCFPTR(x) reinterpret_cast<const float *__restrict__>(__builtin_assume_aligned(x, 16))
#define CFPTR(x) reinterpret_cast<float *__restrict__>(__builtin_assume_aligned(x, 16))

alignas(16) static const double conj_values[] = {1.0f, -1.0f};

using D = hn::CappedTag<float, 4>;

template <class D>
HWY_INLINE auto LoadComplexGroup(D d, const auto *XC, int64_t stride) {
  const auto *X = reinterpret_cast<const MFFTELEMRI *__restrict__>(XC);
  constexpr size_t L = hn::Lanes(d);
  constexpr size_t groups = L / 2;
  if constexpr (groups == 1)
    return hn::Load(d, X);
  else if constexpr (groups == 2) {
    const int64_t istride = stride * 2;
    auto d_complex = hn::FixedTag<MFFTELEMRI, 2>();
    auto a0 = hn::Load(d_complex, X);
    auto a1 = hn::Load(d_complex, X + istride);
    return hn::Combine(d, a1, a0);
  } else
    static_assert(0, "Unsupported lane count");
}

template <class D>
HWY_INLINE void StoreComplexGroup(D d, auto y, auto *__restrict__ YC, int64_t stride) {
  auto *Y = reinterpret_cast<MFFTELEMRI *__restrict__>(YC);
  constexpr size_t L = hn::Lanes(d);
  constexpr size_t groups = L / 2;
  const int64_t istride = stride * 2;
  if constexpr (groups == 1)
    hn::Store(y, d, Y);
  else if constexpr (groups == 2) {
    auto d_complex = hn::FixedTag<MFFTELEMRI, 2>();
    hn::Store(hn::LowerHalf(y), d_complex, Y);
    hn::Store(hn::UpperHalf(d, y), d_complex, Y + istride);
  } else
    static_assert(0, "Unsupported lane count");
}

static inline auto Convert(auto dnew, auto dold, auto &w) {
  if constexpr (std::is_same_v<decltype(dnew), decltype(dold)>) return w;

  constexpr size_t c_new = hn::Lanes(dnew);
  constexpr size_t c_old = hn::Lanes(dold);

  using T_old = hn::TFromD<decltype(dold)>;
  using T_new = hn::TFromD<decltype(dnew)>;  // always same or narrower

  if constexpr (std::is_same_v<T_old, T_new>) {
    if constexpr (c_new == c_old) {
      return w;
    } else if constexpr (c_new == 2 * c_old) {
      return hn::Combine(dnew, w, w);
    } else if constexpr (c_new == 4 * c_old) {
      auto w2 = hn::Combine(dnew, w, w);
      return hn::Combine(dnew, w2, w2);
    } else if constexpr (c_new == 8 * c_old) {
      auto w4 = hn::Combine(dnew, w, w);
      auto w8 = hn::Combine(dnew, w4, w4);
      return hn::Combine(dnew, w8, w8);
    } else if constexpr (2 * c_new == c_old) {
      return hn::LowerHalf(dnew, w);
    } else if constexpr (4 * c_new == c_old) {
      return hn::LowerHalf(dnew, hn::LowerHalf(dold, w));
    } else {
      static_assert(c_new <= 8 * c_old, "Unsupported lane scaling factor");
    }
  } else {
    auto dnew_h = hn::Half<decltype(dnew)>();
    auto w_demoted = hn::DemoteTo(dnew_h, w);  // half size
    return hn::Combine(dnew, w_demoted, w_demoted);
  }
}

static inline void fftr2_kernel(auto d, MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                int64_t bp, int64_t stride, int64_t k, int64_t j, int64_t m,
                                int64_t l, auto w_f) {
  auto v0 = LoadComplexGroup(d, &X[bp + stride * (k + j * m)], stride);
  auto v1 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + l * m)], stride);
  auto d0 = hn::Sub(v0, v1);
  auto y1 = hn::MulComplex(w_f, d0);
  auto y0 = hn::Add(v0, v1);
  StoreComplexGroup(d, y0, &Y[bp + stride * (k + 2 * j * m)], stride);
  StoreComplexGroup(d, y1, &Y[bp + stride * (k + 2 * j * m + m)], stride);
}

void fftr2(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  auto *__restrict__ Y = *YY;
  auto *__restrict__ X = *XX;
  int64_t l = N / 2;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  MFFTELEM *tmp;
  D d;
  auto db = hn::FixedTag<double, 2>();
  auto d2 = hn::Half<decltype(d)>();
  const auto *__restrict__ W = reinterpret_cast<const std::complex<double> *>(COS_SIN_2);
  const auto conj_mask = hn::Load(db, conj_values);
  auto pmim_mask = Convert(d, db, conj_mask);
  if (inverse) pmim_mask = hn::Neg(pmim_mask);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(db, CCDPTR(&W[0]));
    auto w_l = hn::Load(db, CCDPTR(&W[e1 - t - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    auto w_f = Convert(d, db, w);
    for (int64_t j = 0; j < l; j++) {
      for (int64_t k = 0; k < (m - 1); k += 2) fftr2_kernel(d, Y, X, bp, stride, k, j, m, l, w_f);
      if (m & 1) fftr2_kernel(d2, Y, X, bp, stride, m - 1, j, m, l, hn::LowerHalf(w_f));
      w = hn::MulComplex(w, w_l);
      w_f = Convert(d, db, w);
    }
    l >>= 1;
    m <<= 1;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  *XX = Y;
  *YY = X;
}

static inline void fftr3_kernel(auto d, MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                int64_t bp, int64_t stride, int64_t k, int64_t j, int64_t m,
                                int64_t l, auto vc30, auto vc31, auto pmim_mask, auto w_f,
                                auto w2_f) {
  auto v0 = LoadComplexGroup(d, &X[bp + stride * (k + j * m)], stride);
  auto v1 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + l * m)], stride);
  auto v2 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 2 * l * m)], stride);
  auto t0 = hn::Add(v1, v2);
  auto t1 = hn::NegMulAdd(vc30, t0, v0);
  auto t2 = hn::Reverse2(d, hn::Mul(vc31, hn::Sub(v1, v2)));
  t2 = hn::Mul(t2, pmim_mask);
  auto y0 = hn::Add(v0, t0);
  auto y1 = hn::MulComplex(w_f, hn::Add(t1, t2));
  auto y2 = hn::Sub(t1, t2);
  y2 = hn::MulComplex(w2_f, y2);
  StoreComplexGroup(d, y0, &Y[bp + stride * (k + 3 * j * m)], stride);
  StoreComplexGroup(d, y1, &Y[bp + stride * (k + 3 * j * m + m)], stride);
  StoreComplexGroup(d, y2, &Y[bp + stride * (k + 3 * j * m + 2 * m)], stride);
}

void fftr3(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 3;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c30 = 0.5;
  const double c31 = 0.8660254037844386;  // sin(M_PI / 3.0);
  MFFTELEM *tmp;
  D d;
  auto db = hn::FixedTag<double, 2>();
  auto d2 = hn::Half<decltype(d)>();
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_3);
  auto conj_mask = hn::Load(db, conj_values);
  auto pmim_mask = Convert(d, db, conj_mask);
  if (inverse) pmim_mask = hn::Neg(pmim_mask);
  const auto vc30 = hn::Set(d, c30);
  const auto vc31 = hn::Set(d, c31);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(db, CCDPTR(&W[0]));
    auto w_l = hn::Load(db, CCDPTR(&W[e1 - t - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    auto w_f = Convert(d, db, w);
    for (int64_t j = 0; j < l; j++) {
      auto w2_f = hn::MulComplex(w_f, w_f);
      int64_t k;
      for (k = 0; k < (m - 1); k += 2)
        fftr3_kernel(d, Y, X, bp, stride, k, j, m, l, vc30, vc31, pmim_mask, w_f, w2_f);
      // m&1 always true
      fftr3_kernel(d2, Y, X, bp, stride, m - 1, j, m, l, hn::LowerHalf(vc30), hn::LowerHalf(vc31),
                   hn::LowerHalf(pmim_mask), hn::LowerHalf(w_f), hn::LowerHalf(w2_f));
      w = hn::MulComplex(w, w_l);
      w_f = Convert(d, db, w);
    }
    l /= 3;
    m *= 3;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  *XX = Y;
  *YY = X;
}

static inline void fftr4_kernel(auto d, MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                int64_t bp, int64_t stride, int64_t k, int64_t j, int64_t m,
                                int64_t l, auto pmim_mask, auto w_f, auto w2_f, auto w3_f) {
  auto v0 = LoadComplexGroup(d, &X[bp + stride * (k + j * m)], stride);
  auto v1 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + l * m)], stride);
  auto v2 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 2 * l * m)], stride);
  auto v3 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 3 * l * m)], stride);
  auto t0 = hn::Add(v0, v2);
  auto t1 = hn::Sub(v0, v2);
  auto t2 = hn::Add(v1, v3);
  auto t3 = hn::Reverse2(d, hn::Sub(v1, v3));
  t3 = hn::Mul(t3, pmim_mask);
  auto y0 = hn::Add(t0, t2);
  auto y1 = hn::MulComplex(w_f, hn::Add(t1, t3));
  auto y2 = hn::MulComplex(w2_f, hn::Sub(t0, t2));
  auto y3 = hn::MulComplex(w3_f, hn::Sub(t1, t3));
  StoreComplexGroup(d, y0, &Y[bp + stride * (k + 4 * j * m)], stride);
  StoreComplexGroup(d, y1, &Y[bp + stride * (k + 4 * j * m + m)], stride);
  StoreComplexGroup(d, y2, &Y[bp + stride * (k + 4 * j * m + 2 * m)], stride);
  StoreComplexGroup(d, y3, &Y[bp + stride * (k + 4 * j * m + 3 * m)], stride);
}

void fftr4(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N >> 2;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  MFFTELEM *tmp;
  D d;
  constexpr int64_t c_lanes = hn::Lanes(d) / 2;
  auto db = hn::FixedTag<double, 2>();
  auto d2 = hn::Half<decltype(d)>();
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_2);
  const auto conj_mask = hn::Load(db, conj_values);
  auto pmim_mask = Convert(d, db, conj_mask);
  if (inverse) pmim_mask = hn::Neg(pmim_mask);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(db, CCDPTR(&W[0]));
    auto w_l = hn::Load(db, CCDPTR(&W[2 * (e1 - t) - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    auto w_f = Convert(d, db, w);
    for (int64_t j = 0; j < l; j++) {
      auto w2_f = hn::MulComplex(w_f, w_f);
      auto w3_f = hn::MulComplex(w2_f, w_f);
      for (int64_t k = 0; k < (m - 1); k += 2)
        fftr4_kernel(d, Y, X, bp, stride, k, j, m, l, pmim_mask, w_f, w2_f, w3_f);
      if (m & 1)
        fftr4_kernel(d2, Y, X, bp, stride, m - 1, j, m, l, hn::LowerHalf(pmim_mask),
                     hn::LowerHalf(w_f), hn::LowerHalf(w2_f), hn::LowerHalf(w3_f));
      w = hn::MulComplex(w, w_l);
      w_f = Convert(d, db, w);
    }
    l >>= 2;
    m <<= 2;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  *XX = Y;
  *YY = X;
}

static inline void fftr5_kernel(auto d, MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                int64_t bp, int64_t stride, int64_t k, int64_t j, int64_t m,
                                int64_t l, auto vc50, auto vc51, auto vc52, auto vc53,
                                auto pmim_mask, auto w_f, auto w2_f, auto w3_f, auto w4_f) {
  auto v0 = LoadComplexGroup(d, &X[bp + stride * (k + j * m)], stride);
  auto v1 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + l * m)], stride);
  auto v2 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 2 * l * m)], stride);
  auto v3 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 3 * l * m)], stride);
  auto v4 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 4 * l * m)], stride);
  auto t0 = hn::Add(v1, v4);
  auto t1 = hn::Add(v2, v3);
  auto t2 = hn::Mul(vc51, hn::Sub(v1, v4));
  auto t3 = hn::Mul(vc51, hn::Sub(v2, v3));
  auto t4 = hn::Add(t0, t1);
  auto t5 = hn::Mul(vc52, hn::Sub(t0, t1));
  auto t6 = hn::NegMulAdd(vc50, t4, v0);
  auto t7 = hn::Add(t6, t5);
  auto t8 = hn::Sub(t6, t5);
  auto t9 = hn::Reverse2(d, hn::MulAdd(vc53, t3, t2));
  t9 = hn::Mul(t9, pmim_mask);
  auto t10 = hn::Reverse2(d, hn::MulSub(vc53, t2, t3));
  t10 = hn::Mul(t10, pmim_mask);
  auto y0 = hn::Add(v0, t4);
  auto y1 = hn::MulComplex(w_f, hn::Add(t7, t9));
  auto y2 = hn::MulComplex(w2_f, hn::Add(t8, t10));
  auto y3 = hn::MulComplex(w3_f, hn::Sub(t8, t10));
  auto y4 = hn::MulComplex(w4_f, hn::Sub(t7, t9));
  StoreComplexGroup(d, y0, &Y[bp + stride * (k + 5 * j * m)], stride);
  StoreComplexGroup(d, y1, &Y[bp + stride * (k + 5 * j * m + m)], stride);
  StoreComplexGroup(d, y2, &Y[bp + stride * (k + 5 * j * m + 2 * m)], stride);
  StoreComplexGroup(d, y3, &Y[bp + stride * (k + 5 * j * m + 3 * m)], stride);
  StoreComplexGroup(d, y4, &Y[bp + stride * (k + 5 * j * m + 4 * m)], stride);
}

void fftr5(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 5;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c50 = 0.25;
  const double c51 = 0.9510565162951535;  // sin(2.0 * M_PI / 5.0);
  const double c52 = 0.5590169943749475;  // sqrt(5.0) / 4.0;
  const double c53 = 0.6180339887498949;  // sin(M_PI / 5.0) / sin(2.0 * M_PI / 5.0);
  MFFTELEM *tmp;
  D d;
  auto db = hn::FixedTag<double, 2>();
  auto d2 = hn::Half<decltype(d)>();
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_5);
  auto conj_mask = hn::Load(db, conj_values);
  auto pmim_mask = Convert(d, db, conj_mask);
  if (inverse) pmim_mask = hn::Neg(pmim_mask);
  const auto vc50 = hn::Set(d, c50);
  const auto vc51 = hn::Set(d, c51);
  const auto vc52 = hn::Set(d, c52);
  const auto vc53 = hn::Set(d, c53);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(db, CCDPTR(&W[0]));
    auto w_l = hn::Load(db, CCDPTR(&W[e1 - t - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    auto w_f = Convert(d, db, w);
    for (int64_t j = 0; j < l; j++) {
      auto w2_f = hn::MulComplex(w_f, w_f);
      auto w3_f = hn::MulComplex(w2_f, w_f);
      auto w4_f = hn::MulComplex(w2_f, w2_f);
      for (int64_t k = 0; k < (m - 1); k += 2)
        fftr5_kernel(d, Y, X, bp, stride, k, j, m, l, vc50, vc51, vc52, vc53, pmim_mask, w_f, w2_f,
                     w3_f, w4_f);
      // m&1 always true
      fftr5_kernel(d2, Y, X, bp, stride, m - 1, j, m, l, hn::LowerHalf(vc50), hn::LowerHalf(vc51),
                   hn::LowerHalf(vc52), hn::LowerHalf(vc53), hn::LowerHalf(pmim_mask),
                   hn::LowerHalf(w_f), hn::LowerHalf(w2_f), hn::LowerHalf(w3_f),
                   hn::LowerHalf(w4_f));
      w = hn::MulComplex(w, w_l);
      w_f = Convert(d, db, w);
    }
    l /= 5;
    m *= 5;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  *XX = Y;
  *YY = X;
}

static inline void fftr7_kernel(auto d, MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                int64_t bp, int64_t stride, int64_t k, int64_t j, int64_t m,
                                int64_t l, auto vc71, auto vc72, auto vc73, auto vc74, auto vc75,
                                auto vc76, auto vc77, auto vc78, auto pmim_mask, auto w_f,
                                auto w2_f, auto w3_f, auto w4_f, auto w5_f, auto w6_f) {
  auto v0 = LoadComplexGroup(d, &X[bp + stride * (k + j * m)], stride);
  auto v1 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + l * m)], stride);
  auto v2 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 2 * l * m)], stride);
  auto v3 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 3 * l * m)], stride);
  auto v4 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 4 * l * m)], stride);
  auto v5 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 5 * l * m)], stride);
  auto v6 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 6 * l * m)], stride);
  auto t1 = hn::Add(v1, v6);
  auto t2 = hn::Sub(v1, v6);
  auto t3 = hn::Add(v2, v5);
  auto t4 = hn::Sub(v2, v5);
  auto t5 = hn::Add(v3, v4);
  auto t6 = hn::Sub(v3, v4);
  auto t7 = hn::Add(hn::Add(t1, t3), t5);
  auto t8 = hn::Sub(t1, t5);
  auto t9 = hn::Sub(t5, t3);
  auto t10 = hn::Sub(t3, t1);
  auto t11 = hn::Sub(hn::Add(t2, t4), t6);
  auto t12 = hn::Add(t2, t6);
  auto t13 = hn::Neg(hn::Add(t4, t6));
  auto t14 = hn::Sub(t4, t2);
  auto m1 = hn::Mul(vc71, t7);
  auto m2 = hn::Mul(vc72, t8);
  auto m3 = hn::Mul(vc73, t9);
  auto m4 = hn::Mul(vc74, t10);
  auto m5 = hn::Reverse2(d, hn::Mul(vc75, t11));
  m5 = hn::Mul(m5, pmim_mask);
  auto m6 = hn::Reverse2(d, hn::Mul(vc76, t12));
  m6 = hn::Mul(m6, pmim_mask);
  auto m7 = hn::Reverse2(d, hn::Mul(vc77, t13));
  m7 = hn::Mul(m7, pmim_mask);
  auto m8 = hn::Reverse2(d, hn::Mul(vc78, t14));
  m8 = hn::Mul(m8, pmim_mask);
  auto x1 = hn::Sub(v0, m1);
  auto x2 = hn::Add(hn::Add(m2, m3), x1);
  auto x3 = hn::Sub(x1, hn::Add(m2, m4));
  auto x4 = hn::Sub(hn::Add(x1, m4), m3);
  auto x5 = hn::Sub(hn::Add(m5, m6), m7);
  auto x6 = hn::Sub(m5, hn::Add(m6, m8));
  auto x7 = hn::Neg(hn::Add(hn::Add(m5, m7), m8));
  auto y0 = hn::Add(v0, t7);
  auto y1 = hn::MulComplex(w_f, hn::Sub(x2, x5));
  auto y2 = hn::MulComplex(w2_f, hn::Sub(x3, x6));
  auto y3 = hn::MulComplex(w3_f, hn::Sub(x4, x7));
  auto y4 = hn::MulComplex(w4_f, hn::Add(x4, x7));
  auto y5 = hn::MulComplex(w5_f, hn::Add(x3, x6));
  auto y6 = hn::MulComplex(w6_f, hn::Add(x2, x5));
  StoreComplexGroup(d, y0, &Y[bp + stride * (k + 7 * j * m)], stride);
  StoreComplexGroup(d, y1, &Y[bp + stride * (k + 7 * j * m + m)], stride);
  StoreComplexGroup(d, y2, &Y[bp + stride * (k + 7 * j * m + 2 * m)], stride);
  StoreComplexGroup(d, y3, &Y[bp + stride * (k + 7 * j * m + 3 * m)], stride);
  StoreComplexGroup(d, y4, &Y[bp + stride * (k + 7 * j * m + 4 * m)], stride);
  StoreComplexGroup(d, y5, &Y[bp + stride * (k + 7 * j * m + 5 * m)], stride);
  StoreComplexGroup(d, y6, &Y[bp + stride * (k + 7 * j * m + 6 * m)], stride);
}

void fftr7(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 7;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c71 = 0.1666666666666666;    // -(cos(u) + cos(2 * u) + cos(3 * u)) / 3.0;
  const double c72 = 0.7901564685254002;    // (2 * cos(u) - cos(2 * u) - cos(3 * u)) / 3.0;
  const double c73 = 0.05585426728964774;   // (cos(u) - 2 * cos(2 * u) + cos(3 * u)) / 3.0;
  const double c74 = 0.7343022012357524;    // (cos(u) + cos(2 * u) - 2 * cos(3 * u)) / 3.0;
  const double c75 = -0.4409585518440984;   // (sin(u) + sin(2 * u) - sin(3 * u)) / 3.0;
  const double c76 = -0.34087293062393137;  // (2 * sin(u) - sin(2 * u) + sin(3 * u)) / 3.0;
  const double c77 = -0.5339693603377252;   // (-sin(u) + 2 * sin(2 * u) + sin(3 * u)) / 3.0;
  const double c78 = -0.8748422909616567;   // (sin(u) + sin(2 * u) + 2 * sin(3 * u)) / 3.0;
  MFFTELEM *tmp;
  D d;
  constexpr int64_t c_lanes = hn::Lanes(d) / 2;
  auto db = hn::FixedTag<double, 2>();
  auto d2 = hn::Half<decltype(d)>();
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_7);
  auto conj_mask = hn::Load(db, conj_values);
  auto pmim_mask = Convert(d, db, conj_mask);
  if (inverse) pmim_mask = hn::Neg(pmim_mask);
  const auto vc71 = hn::Set(d, c71);
  const auto vc72 = hn::Set(d, c72);
  const auto vc73 = hn::Set(d, c73);
  const auto vc74 = hn::Set(d, c74);
  const auto vc75 = hn::Set(d, c75);
  const auto vc76 = hn::Set(d, c76);
  const auto vc77 = hn::Set(d, c77);
  const auto vc78 = hn::Set(d, c78);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(db, CCDPTR(&W[0]));
    auto w_l = hn::Load(db, CCDPTR(&W[e1 - t - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    auto w_f = Convert(d, db, w);
    for (int64_t j = 0; j < l; j++) {
      auto w2_f = hn::MulComplex(w_f, w_f);
      auto w3_f = hn::MulComplex(w2_f, w_f);
      auto w4_f = hn::MulComplex(w2_f, w2_f);
      auto w5_f = hn::MulComplex(w3_f, w2_f);
      auto w6_f = hn::MulComplex(w3_f, w3_f);
      for (int64_t k = 0; k < (m - 1); k += 2)
        fftr7_kernel(d, Y, X, bp, stride, k, j, m, l, vc71, vc72, vc73, vc74, vc75, vc76, vc77,
                     vc78, pmim_mask, w_f, w2_f, w3_f, w4_f, w5_f, w6_f);
      // m&1 always true
      fftr7_kernel(d2, Y, X, bp, stride, m - 1, j, m, l, hn::LowerHalf(vc71), hn::LowerHalf(vc72),
                   hn::LowerHalf(vc73), hn::LowerHalf(vc74), hn::LowerHalf(vc75),
                   hn::LowerHalf(vc76), hn::LowerHalf(vc77), hn::LowerHalf(vc78),
                   hn::LowerHalf(pmim_mask), hn::LowerHalf(w_f), hn::LowerHalf(w2_f),
                   hn::LowerHalf(w3_f), hn::LowerHalf(w4_f), hn::LowerHalf(w5_f),
                   hn::LowerHalf(w6_f));
      w = hn::MulComplex(w, w_l);
      w_f = Convert(d, db, w);
    }
    l /= 7;
    m *= 7;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  *XX = Y;
  *YY = X;
}

static inline void fftr8_kernel(auto d, MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                int64_t bp, int64_t stride, int64_t k, int64_t j, int64_t m,
                                int64_t l, auto vc81, auto pmim_mask, auto w_f, auto w2_f,
                                auto w3_f, auto w4_f, auto w5_f, auto w6_f, auto w7_f) {
  auto v0 = LoadComplexGroup(d, &X[bp + stride * (k + j * m)], stride);
  auto v1 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + l * m)], stride);
  auto v2 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 2 * l * m)], stride);
  auto v3 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 3 * l * m)], stride);
  auto v4 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 4 * l * m)], stride);
  auto v5 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 5 * l * m)], stride);
  auto v6 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 6 * l * m)], stride);
  auto v7 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 7 * l * m)], stride);
  auto t0 = hn::Add(v0, v4);
  auto t1 = hn::Sub(v0, v4);
  auto t2 = hn::Add(v2, v6);
  auto t3 = hn::Reverse2(d, hn::Sub(v2, v6));
  t3 = hn::Mul(t3, pmim_mask);
  auto t4 = hn::Add(v1, v5);
  auto t5 = hn::Sub(v1, v5);
  auto t6 = hn::Add(v3, v7);
  auto t7 = hn::Sub(v3, v7);
  auto m0 = hn::Add(t0, t2);
  auto m1 = hn::Sub(t0, t2);
  auto m2 = hn::Add(t4, t6);
  auto m3 = hn::Reverse2(d, hn::Sub(t4, t6));
  m3 = hn::Mul(m3, pmim_mask);
  auto m4 = hn::Mul(vc81, hn::Sub(t5, t7));
  auto m5 = hn::Reverse2(d, hn::Mul(vc81, hn::Add(t5, t7)));
  m5 = hn::Mul(m5, pmim_mask);
  auto m6 = hn::Add(t1, m4);
  auto m7 = hn::Sub(t1, m4);
  auto m8 = hn::Add(t3, m5);
  auto m9 = hn::Sub(t3, m5);
  auto y0 = hn::Add(m0, m2);
  auto y1 = hn::MulComplex(w_f, hn::Add(m6, m8));
  auto y2 = hn::MulComplex(w2_f, hn::Add(m1, m3));
  auto y3 = hn::MulComplex(w3_f, hn::Sub(m7, m9));
  auto y4 = hn::MulComplex(w4_f, hn::Sub(m0, m2));
  auto y5 = hn::MulComplex(w5_f, hn::Add(m7, m9));
  auto y6 = hn::MulComplex(w6_f, hn::Sub(m1, m3));
  auto y7 = hn::MulComplex(w7_f, hn::Sub(m6, m8));
  StoreComplexGroup(d, y0, &Y[bp + stride * (k + 8 * j * m)], stride);
  StoreComplexGroup(d, y1, &Y[bp + stride * (k + 8 * j * m + m)], stride);
  StoreComplexGroup(d, y2, &Y[bp + stride * (k + 8 * j * m + 2 * m)], stride);
  StoreComplexGroup(d, y3, &Y[bp + stride * (k + 8 * j * m + 3 * m)], stride);
  StoreComplexGroup(d, y4, &Y[bp + stride * (k + 8 * j * m + 4 * m)], stride);
  StoreComplexGroup(d, y5, &Y[bp + stride * (k + 8 * j * m + 5 * m)], stride);
  StoreComplexGroup(d, y6, &Y[bp + stride * (k + 8 * j * m + 6 * m)], stride);
  StoreComplexGroup(d, y7, &Y[bp + stride * (k + 8 * j * m + 7 * m)], stride);
}

void fftr8(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N >> 3;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c81 = 0.7071067811865476;  // sqrt(2.0) / 2.0;
  D d;
  auto db = hn::FixedTag<double, 2>();
  auto d2 = hn::Half<decltype(d)>();
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_2);
  auto conj_mask = hn::Load(db, conj_values);
  auto pmim_mask = Convert(d, db, conj_mask);
  if (inverse) pmim_mask = hn::Neg(pmim_mask);
  const auto vc81 = hn::Set(d, c81);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(db, CCDPTR(&W[0]));
    auto w_l = hn::Load(db, CCDPTR(&W[3 * (e1 - t) - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    auto w_f = Convert(d, db, w);
    for (int64_t j = 0; j < l; j++) {
      auto w2_f = hn::MulComplex(w_f, w_f);
      auto w3_f = hn::MulComplex(w2_f, w_f);
      auto w4_f = hn::MulComplex(w2_f, w2_f);
      auto w5_f = hn::MulComplex(w3_f, w2_f);
      auto w6_f = hn::MulComplex(w3_f, w3_f);
      auto w7_f = hn::MulComplex(w4_f, w3_f);
      for (int64_t k = 0; k < (m - 1); k += 2)
        fftr8_kernel(d, Y, X, bp, stride, k, j, m, l, vc81, pmim_mask, w_f, w2_f, w3_f, w4_f, w5_f,
                     w6_f, w7_f);
      if (m & 1)
        fftr8_kernel(d2, Y, X, bp, stride, m - 1, j, m, l, hn::LowerHalf(vc81),
                     hn::LowerHalf(pmim_mask), hn::LowerHalf(w_f), hn::LowerHalf(w2_f),
                     hn::LowerHalf(w3_f), hn::LowerHalf(w4_f), hn::LowerHalf(w5_f),
                     hn::LowerHalf(w6_f), hn::LowerHalf(w7_f));
      w = hn::MulComplex(w, w_l);
      w_f = Convert(d, db, w);
    }
    l >>= 3;
    m <<= 3;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  *XX = Y;
  *YY = X;
}

static inline void fftr9_kernel(auto d, MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                int64_t bp, int64_t stride, int64_t k, int64_t j, int64_t m,
                                int64_t l, auto vc90, auto vc91, auto vc93, auto vc94, auto vc95,
                                auto vsu, auto vs2u, auto vs3u, auto vs4u,
                                auto pmim_mask, auto w_f, auto w2_f, auto w3_f, auto w4_f,
                                auto w5_f, auto w6_f, auto w7_f, auto w8_f) {
  auto v0 = LoadComplexGroup(d, &X[bp + stride * (k + j * m)], stride);
  auto v1 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + l * m)], stride);
  auto v2 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 2 * l * m)], stride);
  auto v3 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 3 * l * m)], stride);
  auto v4 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 4 * l * m)], stride);
  auto v5 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 5 * l * m)], stride);
  auto v6 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 6 * l * m)], stride);
  auto v7 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 7 * l * m)], stride);
  auto v8 = LoadComplexGroup(d, &X[bp + stride * (k + j * m + 8 * l * m)], stride);
  auto t1 = hn::Add(v1, v8);
  auto t2 = hn::Add(v2, v7);
  auto t3 = hn::Add(v3, v6);
  auto t4 = hn::Add(v4, v5);
  auto t5 = hn::Add(t1, hn::Add(t2, t4));
  auto t6 = hn::Sub(v1, v8);
  auto t7 = hn::Sub(v7, v2);
  auto t8 = hn::Sub(v3, v6);
  auto t9 = hn::Sub(v4, v5);
  auto t10 = hn::Add(t6, hn::Add(t7, t9));
  auto t11 = hn::Sub(t1, t2);
  auto t12 = hn::Sub(t2, t4);
  auto t13 = hn::Sub(t7, t6);
  auto t14 = hn::Sub(t7, t9);
  auto m0 = hn::Add(v0, hn::Add(t3, t5));
  auto m1 = hn::Mul(vc91, t3);
  auto m2 = hn::Mul(vc90, t5);
  auto t15 = hn::Neg(hn::Add(t12, t11));
  auto m3 = hn::Mul(vc93, t11);
  auto m4 = hn::Mul(vc94, t12);
  auto m5 = hn::Mul(vc95, t15);
  auto s0 = hn::Neg(hn::Add(m3, m4));
  auto s1 = hn::Sub(m5, m4);
  auto m6 = hn::Reverse2(d, hn::Mul(vs3u, t10));
  m6 = hn::Mul(m6, pmim_mask);
  auto m7 = hn::Reverse2(d, hn::Mul(vs3u, t8));
  m7 = hn::Mul(m7, pmim_mask);
  auto t16 = hn::Sub(t14, t13);
  auto m8 = hn::Reverse2(d, hn::Mul(vsu, t13));
  m8 = hn::Mul(m8, pmim_mask);
  auto m9 = hn::Reverse2(d, hn::Mul(vs4u, t14));
  m9 = hn::Mul(m9, pmim_mask);
  auto m10 = hn::Reverse2(d, hn::Mul(vs2u, t16));
  m10 = hn::Mul(m10, pmim_mask);
  auto s2 = hn::Neg(hn::Add(m8, m9));
  auto s3 = hn::Sub(m9, m10);
  auto s4 = hn::Add(m0, hn::Add(m2, m2));
  auto s5 = hn::Sub(s4, m1);
  auto s6 = hn::Add(s4, m2);
  auto s7 = hn::Sub(s5, s0);
  auto s8 = hn::Add(s1, s5);
  auto s9 = hn::Sub(hn::Add(s0, s5), s1);
  auto s10 = hn::Sub(m7, s2);
  auto s11 = hn::Sub(m7, s3);
  auto s12 = hn::Add(m7, hn::Add(s2, s3));
  auto y1 = hn::MulComplex(w_f, hn::Add(s7, s10));
  auto y2 = hn::MulComplex(w2_f, hn::Sub(s8, s11));
  auto y3 = hn::MulComplex(w3_f, hn::Add(s6, m6));
  auto y4 = hn::MulComplex(w4_f, hn::Add(s9, s12));
  auto y5 = hn::MulComplex(w5_f, hn::Sub(s9, s12));
  auto y6 = hn::MulComplex(w6_f, hn::Sub(s6, m6));
  auto y7 = hn::MulComplex(w7_f, hn::Add(s8, s11));
  auto y8 = hn::MulComplex(w8_f, hn::Sub(s7, s10));
  StoreComplexGroup(d, m0, &Y[bp + stride * (k + 9 * j * m)], stride);
  StoreComplexGroup(d, y1, &Y[bp + stride * (k + 9 * j * m + m)], stride);
  StoreComplexGroup(d, y2, &Y[bp + stride * (k + 9 * j * m + 2 * m)], stride);
  StoreComplexGroup(d, y3, &Y[bp + stride * (k + 9 * j * m + 3 * m)], stride);
  StoreComplexGroup(d, y4, &Y[bp + stride * (k + 9 * j * m + 4 * m)], stride);
  StoreComplexGroup(d, y5, &Y[bp + stride * (k + 9 * j * m + 5 * m)], stride);
  StoreComplexGroup(d, y6, &Y[bp + stride * (k + 9 * j * m + 6 * m)], stride);
  StoreComplexGroup(d, y7, &Y[bp + stride * (k + 9 * j * m + 7 * m)], stride);
  StoreComplexGroup(d, y8, &Y[bp + stride * (k + 9 * j * m + 8 * m)], stride);
}

void fftr9(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 9;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c90 = -0.5;
  const double c91 = 3.0 / 2.0;
  const double c93 = 0.766044443118978;    // (2 * cos(u) - cos(2 * u) - cos(4 * u)) / 3.0;
  const double c94 = 0.9396926207859083;   // (cos(u) + cos(2 * u) - 2 * cos(4 * u)) / 3.0;
  const double c95 = -0.1736481776669304;  // (cos(u) - 2 * cos(2 * u) + cos(4 * u)) / 3.0;
  const double su = -0.6427876096865393;    // sin(u);
  const double s2u = -0.984807753012208;    // sin(2 * u);
  const double s3u = 0.8660254037844387;   // sin(3 * u);
  const double s4u = -0.3420201433256689;   // sin(4 * u);
  D d;
  auto db = hn::FixedTag<double, 2>();
  auto d2 = hn::Half<decltype(d)>();  
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_3);
  auto conj_mask = hn::Load(db, conj_values);
  auto pmim_mask = Convert(d, db, conj_mask);
  if (inverse) pmim_mask = hn::Neg(pmim_mask);
  const auto vc90 = hn::Set(d, c90);
  const auto vc91 = hn::Set(d, c91);
  const auto vc93 = hn::Set(d, c93);
  const auto vc94 = hn::Set(d, c94);
  const auto vc95 = hn::Set(d, c95);
  const auto vsu = hn::Set(d, su);
  const auto vs2u = hn::Set(d, s2u);
  const auto vs3u = hn::Set(d, s3u);
  const auto vs4u = hn::Set(d, s4u);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(db, CCDPTR(&W[0]));
    auto w_l = hn::Load(db, CCDPTR(&W[2 * (e1 - t) - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    auto w_f = Convert(d, db, w);
    for (int64_t j = 0; j < l; j++) {
      auto w2_f = hn::MulComplex(w_f, w_f);
      auto w3_f = hn::MulComplex(w2_f, w_f);
      auto w4_f = hn::MulComplex(w2_f, w2_f);
      auto w5_f = hn::MulComplex(w3_f, w2_f);
      auto w6_f = hn::MulComplex(w3_f, w3_f);
      auto w7_f = hn::MulComplex(w4_f, w3_f);
      auto w8_f = hn::MulComplex(w4_f, w4_f);
      for (int64_t k = 0; k < (m-1); k+=2)
        fftr9_kernel(d, Y, X, bp, stride, k, j, m, l, vc90, vc91, vc93, vc94, vc95, vsu, vs2u,
                     vs3u, vs4u, pmim_mask, w_f, w2_f, w3_f, w4_f, w5_f, w6_f, w7_f,
                     w8_f);
      // m&1 always true
      fftr9_kernel(d2, Y, X, bp, stride, m - 1, j, m, l, hn::LowerHalf(vc90), hn::LowerHalf(vc91),
                   hn::LowerHalf(vc93), hn::LowerHalf(vc94), hn::LowerHalf(vc95),
                   hn::LowerHalf(vsu), hn::LowerHalf(vs2u), hn::LowerHalf(vs3u),
                   hn::LowerHalf(vs4u), hn::LowerHalf(pmim_mask), hn::LowerHalf(w_f),
                   hn::LowerHalf(w2_f), hn::LowerHalf(w3_f), hn::LowerHalf(w4_f),
                   hn::LowerHalf(w5_f), hn::LowerHalf(w6_f), hn::LowerHalf(w7_f),
                   hn::LowerHalf(w8_f));
      w = hn::MulComplex(w, w_l);
      w_f = Convert(d, db, w);
    }
    l /= 9;
    m *= 9;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  *XX = Y;
  *YY = X;
}
