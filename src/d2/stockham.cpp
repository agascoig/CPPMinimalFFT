
#include <hwy/highway.h>

#include <complex>
#include <cstdint>

#include "../CPPMinimalFFT.hpp"
#include "../plan.hpp"
#include "../weights.hpp"

namespace hn = hwy::HWY_NAMESPACE;

#define CCDPTR(x) reinterpret_cast<const double *__restrict__>(__builtin_assume_aligned(x, 16))
#define CDPTR(x) reinterpret_cast<double *__restrict__>(__builtin_assume_aligned(x, 16))

alignas(sizeof(double) * 2) static const double conj_values[] = {1.0, -1.0};

using D = hn::FixedTag<double, 2>;

inline void prefetchw(const void *p) {
  __builtin_prefetch(p, 1, 3);  // 1 = write intent, 3 = high temporal locality
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
  const auto *__restrict__ W = reinterpret_cast<const std::complex<double> *>(COS_SIN_2);
  const auto conj_mask = hn::Load(d, conj_values);
  const auto pmim_mask = (inverse) ? hn::Neg(conj_mask) : conj_mask;
  for (int32_t t = 0; t < e1 - 1; t++) {
    auto w = hn::Load(d, CCDPTR(&W[0]));
    auto w_l = hn::Load(d, CCDPTR(&W[e1 - t - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    for (int64_t j = 0; j < l; j++) {
      for (int64_t k = 0; k < m; k++) {
        auto c0 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m)]));
        auto c1 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + l * m)]));
        auto d0 = hn::Sub(c0, c1);
        auto y1 = hn::MulComplex(w, d0);
        auto y0 = hn::Add(c0, c1);
        hn::Store(y0, d, CDPTR(&Y[bp + stride * (k + 2 * j * m)]));
        hn::Store(y1, d, CDPTR(&Y[bp + stride * (k + 2 * j * m + m)]));
      }
      w = hn::MulComplex(w, w_l);
    }
    l >>= 1;
    m <<= 1;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  m = N / 2;
  for (int64_t k = 0; k < m; k++) {
    auto c0 = hn::Load(d, CCDPTR(&X[bp + stride * (k)]));
    auto c1 = hn::Load(d, CCDPTR(&X[bp + stride * (k + m)]));
    auto y0 = hn::Add(c0, c1);
    auto y1 = hn::Sub(c0, c1);
    hn::Store(y0, d, CDPTR(&Y[bp + stride * (k)]));
    hn::Store(y1, d, CDPTR(&Y[bp + stride * (k + m)]));
  }
  *XX = X;
  *YY = Y;
}

void fftr3(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  auto *__restrict__ Y = *YY;
  auto *__restrict__ X = *XX;
  int64_t l = N / 3;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c30 = 0.5;
  const double c31 = 0.8660254037844386;  // sin(M_PI / 3.0);
  MFFTELEM *tmp;
  D d;
  const auto *__restrict__ W =
      reinterpret_cast<const std::complex<double> *__restrict__>(COS_SIN_3);
  const auto conj_mask = hn::Load(d, conj_values);
  const auto pmim_mask = (inverse) ? hn::Neg(conj_mask) : conj_mask;
  const auto vc30 = hn::Set(d, c30);
  const auto vc31 = hn::Set(d, c31);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(d, CCDPTR(&W[0]));
    auto w_l = hn::Load(d, CCDPTR(&W[e1 - t - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    for (int64_t j = 0; j < l; j++) {
      auto w2 = hn::MulComplex(w, w);
      for (int64_t k = 0; k < m; k++) {
        auto c0 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m)]));
        auto c1 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + l * m)]));
        auto c2 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 2 * l * m)]));
        auto d0 = hn::Add(c1, c2);
        auto d1 = hn::NegMulAdd(vc30, d0, c0);
        auto d2 = hn::Reverse2(d, hn::Mul(vc31, hn::Sub(c1, c2)));
        d2 = hn::Mul(d2, pmim_mask);
        auto y0 = hn::Add(c0, d0);
        auto y1 = hn::MulComplex(w, hn::Add(d1, d2));
        auto y2 = hn::Sub(d1, d2);
        y2 = hn::MulComplex(w2, y2);
        hn::Store(y0, d, CDPTR(&Y[bp + stride * (k + 3 * j * m)]));
        hn::Store(y1, d, CDPTR(&Y[bp + stride * (k + 3 * j * m + m)]));
        hn::Store(y2, d, CDPTR(&Y[bp + stride * (k + 3 * j * m + 2 * m)]));
      }
      w = hn::MulComplex(w, w_l);
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

void fftr4(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  auto *__restrict__ Y = *YY;
  auto *__restrict__ X = *XX;
  int64_t l = N >> 2;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  MFFTELEM *tmp;
  D d;
  const auto *__restrict__ W = reinterpret_cast<const std::complex<double> *>(COS_SIN_2);
  const auto conj_mask = hn::Load(d, conj_values);
  const auto pmim_mask = (inverse) ? hn::Neg(conj_mask) : conj_mask;
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(d, CCDPTR(&W[0]));
    auto w_l = hn::Load(d, CCDPTR(&W[2 * (e1 - t) - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    for (int64_t j = 0; j < l; j++) {
      auto w2 = hn::MulComplex(w, w);
      auto w3 = hn::MulComplex(w2, w);
      for (int64_t k = 0; k < m; k++) {
        auto c0 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m)]));
        auto c1 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + m * l)]));
        auto c2 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 2 * m * l)]));
        auto c3 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 3 * m * l)]));
        auto d0 = hn::Add(c0, c2);
        auto d1 = hn::Sub(c0, c2);
        auto d2 = hn::Add(c1, c3);
        auto d3 = hn::Reverse2(d, hn::Sub(c1, c3));
        d3 = hn::Mul(d3, pmim_mask);
        auto y0 = hn::Add(d0, d2);
        auto y1 = hn::MulComplex(w, hn::Add(d1, d3));
        auto y2 = hn::MulComplex(w2, hn::Sub(d0, d2));
        auto y3 = hn::MulComplex(w3, hn::Sub(d1, d3));
        hn::Store(y0, d, CDPTR(&Y[bp + stride * (k + 4 * j * m)]));
        hn::Store(y1, d, CDPTR(&Y[bp + stride * (k + 4 * j * m + m)]));
        hn::Store(y2, d, CDPTR(&Y[bp + stride * (k + 4 * j * m + 2 * m)]));
        hn::Store(y3, d, CDPTR(&Y[bp + stride * (k + 4 * j * m + 3 * m)]));
      }
      w = hn::MulComplex(w, w_l);
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

void fftr5(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  auto *__restrict__ Y = *YY;
  auto *__restrict__ X = *XX;
  int64_t l = N / 5;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c50 = 0.25;
  const double c51 = 0.9510565162951535;  // sin(2.0 * M_PI / 5.0);
  const double c52 = 0.5590169943749475;  // sqrt(5.0) / 4.0;
  const double c53 = 0.6180339887498949;  // sin(M_PI / 5.0) / sin(2.0 * M_PI / 5.0);
  MFFTELEM *tmp;
  D d;
  const auto *__restrict__ W = reinterpret_cast<const std::complex<double> *>(COS_SIN_5);
  const auto conj_mask = hn::Load(d, conj_values);
  const auto pmim_mask = (inverse) ? hn::Neg(conj_mask) : conj_mask;
  const auto vc50 = hn::Set(d, c50);
  const auto vc51 = hn::Set(d, c51);
  const auto vc52 = hn::Set(d, c52);
  const auto vc53 = hn::Set(d, c53);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(d, CCDPTR(&W[0]));
    auto w_l = hn::Load(d, CCDPTR(&W[e1 - t - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    for (int64_t j = 0; j < l; j++) {
      auto w2 = hn::MulComplex(w, w);
      auto w3 = hn::MulComplex(w2, w);
      auto w4 = hn::MulComplex(w2, w2);
      for (int64_t k = 0; k < m; k++) {
        auto c0 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m)]));
        auto c1 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + l * m)]));
        auto c2 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 2 * l * m)]));
        auto c3 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 3 * l * m)]));
        auto c4 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 4 * l * m)]));
        auto d0 = hn::Add(c1, c4);
        auto d1 = hn::Add(c2, c3);
        auto d2 = hn::Mul(vc51, hn::Sub(c1, c4));
        auto d3 = hn::Mul(vc51, hn::Sub(c2, c3));
        auto d4 = hn::Add(d0, d1);
        auto d5 = hn::Mul(vc52, hn::Sub(d0, d1));
        auto d6 = hn::NegMulAdd(vc50, d4, c0);
        auto d7 = hn::Add(d6, d5);
        auto d8 = hn::Sub(d6, d5);
        auto d9 = hn::MulAdd(vc53, d3, d2);
        d9 = hn::Reverse2(d, d9);
        d9 = hn::Mul(d9, pmim_mask);
        auto d10 = hn::MulSub(vc53, d2, d3);
        d10 = hn::Reverse2(d, d10);
        d10 = hn::Mul(d10, pmim_mask);
        auto y0 = hn::Add(c0, d4);
        auto y1 = hn::MulComplex(w, hn::Add(d7, d9));
        auto y2 = hn::MulComplex(w2, hn::Add(d8, d10));
        auto y3 = hn::MulComplex(w3, hn::Sub(d8, d10));
        auto y4 = hn::MulComplex(w4, hn::Sub(d7, d9));
        hn::Store(y0, d, CDPTR(&Y[bp + stride * (k + 5 * j * m)]));
        hn::Store(y1, d, CDPTR(&Y[bp + stride * (k + 5 * j * m + m)]));
        hn::Store(y2, d, CDPTR(&Y[bp + stride * (k + 5 * j * m + 2 * m)]));
        hn::Store(y3, d, CDPTR(&Y[bp + stride * (k + 5 * j * m + 3 * m)]));
        hn::Store(y4, d, CDPTR(&Y[bp + stride * (k + 5 * j * m + 4 * m)]));
      }
      w = hn::MulComplex(w, w_l);
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

void fftr7(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  auto *__restrict__ Y = *YY;
  auto *__restrict__ X = *XX;
  int64_t l = N / 7;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c71 = 0.1666666666666666;    // -(cos(u) + cos(2 * u) + cos(3 * u)) / 3.0;
  const double c72 = 0.7901564685254002;    // (2 * cos(u) - cos(2 * u) - cos(3 * u)) / 3.0;
  const double c73 = 0.05585426728964774;   // (cos(u) - 2 * cos(2 * u) + cos(3 * u)) / 3.0;
  const double c74 = 0.7343022012357524;    // (cos(u) + cos(2 * u) - 2 * cos(3 * u)) / 3.0;
  const double c75 = -0.4409585518440984;   // -(sin(u) + sin(2 * u) - sin(3 * u)) / 3.0;
  const double c76 = -0.34087293062393137;  // -(2 * sin(u) - sin(2 * u) + sin(3 * u)) / 3.0;
  const double c77 = -0.5339693603377252;   // -(-sin(u) + 2 * sin(2 * u) + sin(3 * u)) / 3.0;
  const double c78 = -0.8748422909616567;   // -(sin(u) + sin(2 * u) + 2 * sin(3 * u)) / 3.0;
  MFFTELEM *tmp;
  D d;
  const auto *W = reinterpret_cast<const std::complex<double> *>(COS_SIN_7);
  const auto conj_mask = hn::Load(d, conj_values);
  const auto pmim_mask = (inverse) ? hn::Neg(conj_mask) : conj_mask;
  auto vc71 = hn::Set(d, c71);
  auto vc72 = hn::Set(d, c72);
  auto vc73 = hn::Set(d, c73);
  auto vc74 = hn::Set(d, c74);
  auto vc75 = hn::Set(d, c75);
  auto vc76 = hn::Set(d, c76);
  auto vc77 = hn::Set(d, c77);
  auto vc78 = hn::Set(d, c78);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(d, CCDPTR(&W[0]));
    auto w_l = hn::Load(d, CCDPTR(&W[e1 - t - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    for (int64_t j = 0; j < l; j++) {
      auto w2 = hn::MulComplex(w, w);
      auto w3 = hn::MulComplex(w2, w);
      auto w4 = hn::MulComplex(w2, w2);
      auto w5 = hn::MulComplex(w4, w);
      auto w6 = hn::MulComplex(w3, w3);
      for (int64_t k = 0; k < m; k++) {
        auto c0 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m)]));
        auto c1 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + l * m)]));
        auto c2 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 2 * l * m)]));
        auto c3 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 3 * l * m)]));
        auto c4 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 4 * l * m)]));
        auto c5 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 5 * l * m)]));
        auto c6 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 6 * l * m)]));
        auto a1 = hn::Add(c1, c6);
        auto a2 = hn::Sub(c1, c6);
        auto a3 = hn::Add(c2, c5);
        auto a4 = hn::Sub(c2, c5);
        auto a5 = hn::Add(c3, c4);
        auto a6 = hn::Sub(c3, c4);
        auto a7 = hn::Add(a1, hn::Add(a3, a5));
        auto a8 = hn::Sub(a1, a5);
        auto a9 = hn::Sub(a5, a3);
        auto a10 = hn::Sub(a3, a1);
        auto a11 = hn::Add(a2, hn::Sub(a4, a6));
        auto a12 = hn::Add(a2, a6);
        auto a13 = hn::Neg(hn::Add(a4, a6));
        auto a14 = hn::Sub(a4, a2);
        auto m1 = hn::Mul(vc71, a7);
        auto m2 = hn::Mul(vc72, a8);
        auto m3 = hn::Mul(vc73, a9);
        auto m4 = hn::Mul(vc74, a10);
        auto m5 = hn::Mul(vc75, a11);
        m5 = hn::Reverse2(d, m5);
        m5 = hn::Mul(m5, pmim_mask);
        auto m6 = hn::Mul(vc76, a12);
        m6 = hn::Reverse2(d, m6);
        m6 = hn::Mul(m6, pmim_mask);
        auto m7 = hn::Mul(vc77, a13);
        m7 = hn::Reverse2(d, m7);
        m7 = hn::Mul(m7, pmim_mask);
        auto m8 = hn::Mul(vc78, a14);
        m8 = hn::Reverse2(d, m8);
        m8 = hn::Mul(m8, pmim_mask);
        auto x1 = hn::Sub(c0, m1);
        auto x2 = hn::Add(x1, hn::Add(m2, m3));
        auto x3 = hn::Sub(x1, hn::Add(m2, m4));
        auto x4 = hn::Sub(hn::Add(x1, m4), m3);
        auto x5 = hn::Sub(hn::Add(m5, m6), m7);
        auto x6 = hn::Sub(m5, hn::Add(m6, m8));
        auto x7 = hn::Neg(hn::Add(m5, hn::Add(m7, m8)));
        auto y0 = hn::Add(c0, a7);
        auto y1 = hn::MulComplex(w, hn::Sub(x2, x5));
        auto y2 = hn::MulComplex(w2, hn::Sub(x3, x6));
        auto y3 = hn::MulComplex(w3, hn::Sub(x4, x7));
        auto y4 = hn::MulComplex(w4, hn::Add(x4, x7));
        auto y5 = hn::MulComplex(w5, hn::Add(x3, x6));
        auto y6 = hn::MulComplex(w6, hn::Add(x2, x5));
        hn::Store(y0, d, CDPTR(&Y[bp + stride * (k + 7 * j * m)]));
        hn::Store(y1, d, CDPTR(&Y[bp + stride * (k + 7 * j * m + m)]));
        hn::Store(y2, d, CDPTR(&Y[bp + stride * (k + 7 * j * m + 2 * m)]));
        hn::Store(y3, d, CDPTR(&Y[bp + stride * (k + 7 * j * m + 3 * m)]));
        hn::Store(y4, d, CDPTR(&Y[bp + stride * (k + 7 * j * m + 4 * m)]));
        hn::Store(y5, d, CDPTR(&Y[bp + stride * (k + 7 * j * m + 5 * m)]));
        hn::Store(y6, d, CDPTR(&Y[bp + stride * (k + 7 * j * m + 6 * m)]));
      }
      w = hn::MulComplex(w, w_l);
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

void fftr8(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  auto *__restrict__ Y = *YY;
  auto *__restrict__ X = *XX;
  int64_t l = N >> 3;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c81 = 0.7071067811865476;  // sqrt(2.0) / 2.0;
  MFFTELEM *tmp;
  D d;
  const auto *__restrict__ W =
      reinterpret_cast<const std::complex<double> *__restrict__>(COS_SIN_2);
  const auto conj_mask = hn::Load(d, conj_values);
  const auto pmim_mask = (inverse) ? hn::Neg(conj_mask) : conj_mask;
  const auto vc81 = hn::Set(d, c81);
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(d, CCDPTR(&W[0]));
    auto w_l = hn::Load(d, CCDPTR(&W[3 * (e1 - t) - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    for (int64_t j = 0; j < l; j++) {
      auto w2 = hn::MulComplex(w, w);
      auto w3 = hn::MulComplex(w2, w);
      auto w4 = hn::MulComplex(w2, w2);
      auto w5 = hn::MulComplex(w4, w);
      auto w6 = hn::MulComplex(w3, w3);
      auto w7 = hn::MulComplex(w4, w3);
      for (int64_t k = 0; k < m; k++) {
        auto c0 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m)]));
        auto c1 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + l * m)]));
        auto c2 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 2 * l * m)]));
        auto c3 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 3 * l * m)]));
        auto c4 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 4 * l * m)]));
        auto c5 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 5 * l * m)]));
        auto c6 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 6 * l * m)]));
        auto c7 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 7 * l * m)]));
        auto d0 = hn::Add(c0, c4);
        auto d1 = hn::Sub(c0, c4);
        auto d2 = hn::Add(c2, c6);
        auto d3 = hn::Sub(c2, c6);
        d3 = hn::Reverse2(d, d3);
        d3 = hn::Mul(d3, pmim_mask);
        auto d4 = hn::Add(c1, c5);
        auto d5 = hn::Sub(c1, c5);
        auto d6 = hn::Add(c3, c7);
        auto d7 = hn::Sub(c3, c7);
        auto m0 = hn::Add(d0, d2);
        auto m1 = hn::Sub(d0, d2);
        auto m2 = hn::Add(d4, d6);
        auto m3 = hn::Sub(d4, d6);
        m3 = hn::Reverse2(d, m3);
        m3 = hn::Mul(m3, pmim_mask);
        auto m4 = hn::Mul(vc81, hn::Sub(d5, d7));
        auto m5 = hn::Mul(vc81, hn::Add(d5, d7));
        m5 = hn::Reverse2(d, m5);
        m5 = hn::Mul(m5, pmim_mask);
        auto m6 = hn::Add(d1, m4);
        auto m7 = hn::Sub(d1, m4);
        auto m8 = hn::Add(d3, m5);
        auto m9 = hn::Sub(d3, m5);
        auto y0 = hn::Add(m0, m2);
        auto y1 = hn::MulComplex(w, hn::Add(m6, m8));
        auto y2 = hn::MulComplex(w2, hn::Add(m1, m3));
        auto y3 = hn::MulComplex(w3, hn::Sub(m7, m9));
        auto y4 = hn::MulComplex(w4, hn::Sub(m0, m2));
        auto y5 = hn::MulComplex(w5, hn::Add(m7, m9));
        auto y6 = hn::MulComplex(w6, hn::Sub(m1, m3));
        auto y7 = hn::MulComplex(w7, hn::Sub(m6, m8));
        hn::Store(y0, d, CDPTR(&Y[bp + stride * (k + 8 * j * m)]));
        hn::Store(y1, d, CDPTR(&Y[bp + stride * (k + 8 * j * m + m)]));
        hn::Store(y2, d, CDPTR(&Y[bp + stride * (k + 8 * j * m + 2 * m)]));
        hn::Store(y3, d, CDPTR(&Y[bp + stride * (k + 8 * j * m + 3 * m)]));
        hn::Store(y4, d, CDPTR(&Y[bp + stride * (k + 8 * j * m + 4 * m)]));
        hn::Store(y5, d, CDPTR(&Y[bp + stride * (k + 8 * j * m + 5 * m)]));
        hn::Store(y6, d, CDPTR(&Y[bp + stride * (k + 8 * j * m + 6 * m)]));
        hn::Store(y7, d, CDPTR(&Y[bp + stride * (k + 8 * j * m + 7 * m)]));
      }
      w = hn::MulComplex(w, w_l);
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

void fftr9(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  auto *__restrict__ Y = *YY;
  auto *__restrict__ X = *XX;
  int64_t l = N / 9;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c90 = -0.5;
  const double c91 = 3.0 / 2.0;
  const double c93 = 0.766044443118978;    // (2 * cos(u) - cos(2 * u) - cos(4 * u)) / 3.0;
  const double c94 = 0.9396926207859083;   // (cos(u) + cos(2 * u) - 2 * cos(4 * u)) / 3.0;
  const double c95 = -0.1736481776669304;  // (cos(u) - 2 * cos(2 * u) + cos(4 * u)) / 3.0;
  const double su = -0.6427876096865393;   // -sin(u);
  const double s2u = -0.984807753012208;   // -sin(2 * u);
  const double s3u = 0.8660254037844387;   // sin(3 * u);
  const double s4u = -0.3420201433256689;  // -sin(4 * u);
  MFFTELEM *tmp;
  D d;
  auto vc90 = hn::Set(d, c90);
  auto vc91 = hn::Set(d, c91);
  auto vc93 = hn::Set(d, c93);
  auto vc94 = hn::Set(d, c94);
  auto vc95 = hn::Set(d, c95);
  auto vsu = hn::Set(d, su);
  auto vs2u = hn::Set(d, s2u);
  auto vs3u = hn::Set(d, s3u);
  auto vs4u = hn::Set(d, s4u);
  const auto *__restrict__ W = reinterpret_cast<const std::complex<double> *>(COS_SIN_3);
  const auto conj_mask = hn::Load(d, conj_values);
  const auto pmim_mask = (inverse) ? hn::Neg(conj_mask) : conj_mask;
  for (int32_t t = 0; t < e1; t++) {
    auto w = hn::Load(d, CCDPTR(&W[0]));
    auto w_l = hn::Load(d, CCDPTR(&W[2 * (e1 - t) - 1]));
    if (inverse) w_l = hn::Mul(w_l, conj_mask);
    for (int64_t j = 0; j < l; j++) {
      auto w2 = hn::MulComplex(w, w);
      auto w3 = hn::MulComplex(w2, w);
      auto w4 = hn::MulComplex(w2, w2);
      auto w5 = hn::MulComplex(w4, w);
      auto w6 = hn::MulComplex(w3, w3);
      auto w7 = hn::MulComplex(w4, w3);
      auto w8 = hn::MulComplex(w4, w4);
      for (int64_t k = 0; k < m; k++) {
        auto c0 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m)]));
        auto c1 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + l * m)]));
        auto c2 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 2 * l * m)]));
        auto c3 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 3 * l * m)]));
        auto c4 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 4 * l * m)]));
        auto c5 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 5 * l * m)]));
        auto c6 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 6 * l * m)]));
        auto c7 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 7 * l * m)]));
        auto c8 = hn::Load(d, CCDPTR(&X[bp + stride * (k + j * m + 8 * l * m)]));
        auto t1 = hn::Add(c1, c8);
        auto t2 = hn::Add(c2, c7);
        auto t3 = hn::Add(c3, c6);
        auto t4 = hn::Add(c4, c5);
        auto t5 = hn::Add(t1, hn::Add(t2, t4));
        auto t6 = hn::Sub(c1, c8);
        auto t7 = hn::Sub(c7, c2);
        auto t8 = hn::Sub(c3, c6);
        auto t9 = hn::Sub(c4, c5);
        auto t10 = hn::Add(t6, hn::Add(t7, t9));
        auto t11 = hn::Sub(t1, t2);
        auto t12 = hn::Sub(t2, t4);
        auto t13 = hn::Sub(t7, t6);
        auto t14 = hn::Sub(t7, t9);
        auto m0 = hn::Add(c0, hn::Add(t3, t5));
        auto m1 = hn::Mul(vc91, t3);
        auto m2 = hn::Mul(t5, vc90);
        auto t15 = hn::Neg(hn::Add(t11, t12));
        auto m3 = hn::Mul(vc93, t11);
        auto m4 = hn::Mul(vc94, t12);
        auto m5 = hn::Mul(vc95, t15);
        auto s0 = hn::Neg(hn::Add(m3, m4));
        auto s1 = hn::Sub(m5, m4);
        auto m6 = hn::Mul(vs3u, t10);
        m6 = hn::Reverse2(d, m6);
        m6 = hn::Mul(m6, pmim_mask);
        auto m7 = hn::Mul(vs3u, t8);
        m7 = hn::Reverse2(d, m7);
        m7 = hn::Mul(m7, pmim_mask);
        auto t16 = hn::Sub(t14, t13);
        auto m8 = hn::Mul(vsu, t13);
        m8 = hn::Reverse2(d, m8);
        m8 = hn::Mul(m8, pmim_mask);
        auto m9 = hn::Mul(vs4u, t14);
        m9 = hn::Reverse2(d, m9);
        m9 = hn::Mul(m9, pmim_mask);
        auto m10 = hn::Mul(vs2u, t16);
        m10 = hn::Reverse2(d, m10);
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
        auto y1 = hn::MulComplex(w, hn::Add(s7, s10));
        auto y2 = hn::MulComplex(w2, hn::Sub(s8, s11));
        auto y3 = hn::MulComplex(w3, hn::Add(s6, m6));
        auto y4 = hn::MulComplex(w4, hn::Add(s9, s12));
        auto y5 = hn::MulComplex(w5, hn::Sub(s9, s12));
        auto y6 = hn::MulComplex(w6, hn::Sub(s6, m6));
        auto y7 = hn::MulComplex(w7, hn::Add(s8, s11));
        auto y8 = hn::MulComplex(w8, hn::Sub(s7, s10));
        hn::Store(m0, d, CDPTR(&Y[bp + stride * (k + 9 * j * m)]));
        hn::Store(y1, d, CDPTR(&Y[bp + stride * (k + 9 * j * m + m)]));
        hn::Store(y2, d, CDPTR(&Y[bp + stride * (k + 9 * j * m + 2 * m)]));
        hn::Store(y3, d, CDPTR(&Y[bp + stride * (k + 9 * j * m + 3 * m)]));
        hn::Store(y4, d, CDPTR(&Y[bp + stride * (k + 9 * j * m + 4 * m)]));
        hn::Store(y5, d, CDPTR(&Y[bp + stride * (k + 9 * j * m + 5 * m)]));
        hn::Store(y6, d, CDPTR(&Y[bp + stride * (k + 9 * j * m + 6 * m)]));
        hn::Store(y7, d, CDPTR(&Y[bp + stride * (k + 9 * j * m + 7 * m)]));
        hn::Store(y8, d, CDPTR(&Y[bp + stride * (k + 9 * j * m + 8 * m)]));
      }
      w = hn::MulComplex(w, w_l);
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