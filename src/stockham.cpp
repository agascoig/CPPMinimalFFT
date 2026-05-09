
#include <complex>
#include <cstdint>

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"
#include "weights.hpp"

template <bool Inverse>
void fftr2(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  int64_t l = N / 2;
  int64_t m = 1;
  MFFTELEM* tmp;
  const std::complex<double>* __restrict__ W =
      reinterpret_cast<const std::complex<double>*>(COS_SIN_2);
  for (int32_t t = 0; t < e1; t++) {
    auto w = W[0];
    auto w_l = W[e1 - t - 1];
    if constexpr (Inverse) {
      w_l = conj(w_l);
    }
    for (int64_t j = 0; j < l; j++) {
      for (int64_t k = 0; k < m; k++) {
        auto c0 = X[bp + stride * (k + j * m)];
        auto c1 = X[bp + stride * (k + j * m + l * m)];
        Y[bp + stride * (k + 2 * j * m)] = c0 + c1;
        Y[bp + stride * (k + 2 * j * m + m)] = ((MFFTELEM)w * (c0 - c1));
      }
      w = w * w_l;
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

template <bool Inverse>
void fftr3(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  int64_t l = N / 3;
  int64_t m = 1;
  const double c30 = 0.5;
  const double c31 = 0.8660254037844386;  // sin(M_PI / 3.0);
  MFFTELEM* tmp;
  const std::complex<double>* __restrict__ W =
      reinterpret_cast<const std::complex<double>*>(COS_SIN_3);
  for (int32_t t = 0; t < e1; t++) {
    auto w = W[0];
    auto w_l = W[e1 - t - 1];
    if constexpr (Inverse) {
      w_l = conj(w_l);
    }
    for (int64_t j = 0; j < l; j++) {
      for (int64_t k = 0; k < m; k++) {
        auto c0 = X[bp + stride * (k + j * m)];
        auto c1 = X[bp + stride * (k + j * m + l * m)];
        auto c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        auto d0 = c1 + c2;
        auto d1 = c0 - (MFFTELEMRI)c30 * d0;
        auto d2 = times_pmim<Inverse>(((MFFTELEMRI)c31 * (c1 - c2)));
        auto w2 = w * w;
        Y[bp + stride * (k + 3 * j * m)] = (c0 + d0);
        Y[bp + stride * (k + 3 * j * m + m)] = ((MFFTELEM)w * (d1 + d2));
        Y[bp + stride * (k + 3 * j * m + 2 * m)] = ((MFFTELEM)w2 * (d1 - d2));
      }
      w = w * w_l;
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

template <bool Inverse>           
void fftr4(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  int64_t l = N >> 2;
  int64_t m = 1;
  MFFTELEM* tmp;
  const std::complex<double>* __restrict__ W =
      reinterpret_cast<const std::complex<double>*>(COS_SIN_2);
  for (int32_t t = 0; t < e1; t++) {
    auto w = W[0];
    auto w_l = W[2 * (e1 - t) - 1];
    if constexpr (Inverse) {
      w_l = conj(w_l);
    }
    for (int64_t j = 0; j < l; j++) {
      auto w2 = w * w;
      auto w3 = w2 * w;
      for (int64_t k = 0; k < m; k++) {
        auto c0 = X[bp + stride * (k + j * m)];
        auto c1 = X[bp + stride * (k + j * m + m * l)];
        auto c2 = X[bp + stride * (k + j * m + 2 * m * l)];
        auto c3 = X[bp + stride * (k + j * m + 3 * m * l)];
        auto d0 = c0 + c2;
        auto d1 = c0 - c2;
        auto d2 = c1 + c3;
        auto d3 = times_pmim<Inverse>(c1 - c3);
        Y[bp + stride * (k + 4 * j * m)] = (d0 + d2);
        Y[bp + stride * (k + 4 * j * m + m)] = ((MFFTELEM)w * (d1 + d3));
        Y[bp + stride * (k + 4 * j * m + 2 * m)] = ((MFFTELEM)w2 * (d0 - d2));
        Y[bp + stride * (k + 4 * j * m + 3 * m)] = ((MFFTELEM)w3 * (d1 - d3));
      }
      w = w * w_l;
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

template <bool Inverse>
void fftr5(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  int64_t l = N / 5;
  int64_t m = 1;
  const double c50 = 0.25;
  const double c51 = 0.9510565162951535;  // sin(2.0 * M_PI / 5.0);
  const double c52 = 0.5590169943749475;  // sqrt(5.0) / 4.0;
  const double c53 = 0.6180339887498949;  // sin(M_PI / 5.0) / sin(2.0 * M_PI / 5.0);
  MFFTELEM* tmp;
  const std::complex<double>* __restrict__ W =
      reinterpret_cast<const std::complex<double>*>(COS_SIN_5);
  for (int32_t t = 0; t < e1; t++) {
    auto w = W[0];
    auto w_l = W[e1 - t - 1];
    if constexpr (Inverse) {
      w_l = conj(w_l);
    }
    for (int64_t j = 0; j < l; j++) {
      auto w2 = w * w;
      auto w3 = w2 * w;
      auto w4 = w2 * w2;
      for (int64_t k = 0; k < m; k++) {
        auto c0 = X[bp + stride * (k + j * m)];
        auto c1 = X[bp + stride * (k + j * m + l * m)];
        auto c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        auto c3 = X[bp + stride * (k + j * m + 3 * l * m)];
        auto c4 = X[bp + stride * (k + j * m + 4 * l * m)];
        auto d0 = c1 + c4;
        auto d1 = c2 + c3;
        auto d2 = (MFFTELEMRI)c51 * (c1 - c4);
        auto d3 = (MFFTELEMRI)c51 * (c2 - c3);
        auto d4 = d0 + d1;
        auto d5 = (MFFTELEMRI)c52 * (d0 - d1);
        auto d6 = c0 - (MFFTELEMRI)c50 * d4;
        auto d7 = d6 + d5;
        auto d8 = d6 - d5;
        auto d9 = times_pmim<Inverse>(d2 + (MFFTELEMRI)c53 * d3);
        auto d10 = times_pmim<Inverse>((MFFTELEMRI)c53 * d2 - d3);
        Y[bp + stride * (k + 5 * j * m)] = (c0 + d4);
        Y[bp + stride * (k + 5 * j * m + m)] = ((MFFTELEM)w * (d7 + d9));
        Y[bp + stride * (k + 5 * j * m + 2 * m)] = ((MFFTELEM)w2 * (d8 + d10));
        Y[bp + stride * (k + 5 * j * m + 3 * m)] = ((MFFTELEM)w3 * (d8 - d10));
        Y[bp + stride * (k + 5 * j * m + 4 * m)] = ((MFFTELEM)w4 * (d7 - d9));
      }
      w = w * w_l;
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

template <bool Inverse>
void fftr7(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  int64_t l = N / 7;
  int64_t m = 1;
  const double c71 = 0.1666666666666666;   // -(cos(u) + cos(2 * u) + cos(3 * u)) / 3.0;
  const double c72 = 0.7901564685254002;   // (2 * cos(u) - cos(2 * u) - cos(3 * u)) / 3.0;
  const double c73 = 0.05585426728964774;  // (cos(u) - 2 * cos(2 * u) + cos(3 * u)) / 3.0;
  const double c74 = 0.7343022012357524;   // (cos(u) + cos(2 * u) - 2 * cos(3 * u)) / 3.0;
  const double c75 = 0.4409585518440984;   // (sin(u) + sin(2 * u) - sin(3 * u)) / 3.0;
  const double c76 = 0.34087293062393137;  // (2 * sin(u) - sin(2 * u) + sin(3 * u)) / 3.0;
  const double c77 = 0.5339693603377252;   // (-sin(u) + 2 * sin(2 * u) + sin(3 * u)) / 3.0;
  const double c78 = 0.8748422909616567;   // (sin(u) + sin(2 * u) + 2 * sin(3 * u)) / 3.0;
  MFFTELEM* tmp;
  const std::complex<double>* __restrict__ W =
      reinterpret_cast<const std::complex<double>*>(COS_SIN_7);
  for (int32_t t = 0; t < e1; t++) {
    auto w = W[0];
    auto w_l = W[e1 - t - 1];
    if constexpr (Inverse) {
      w_l = conj(w_l);
    }
    for (int64_t j = 0; j < l; j++) {
      auto w2 = w * w;
      auto w3 = w2 * w;
      auto w4 = w2 * w2;
      auto w5 = w4 * w;
      auto w6 = w3 * w3;
      for (int64_t k = 0; k < m; k++) {
        auto c0 = X[bp + stride * (k + j * m)];
        auto c1 = X[bp + stride * (k + j * m + l * m)];
        auto c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        auto c3 = X[bp + stride * (k + j * m + 3 * l * m)];
        auto c4 = X[bp + stride * (k + j * m + 4 * l * m)];
        auto c5 = X[bp + stride * (k + j * m + 5 * l * m)];
        auto c6 = X[bp + stride * (k + j * m + 6 * l * m)];
        auto a1 = c1 + c6;
        auto a2 = c1 - c6;
        auto a3 = c2 + c5;
        auto a4 = c2 - c5;
        auto a5 = c3 + c4;
        auto a6 = c3 - c4;
        auto a7 = a1 + a3 + a5;
        auto a8 = a1 - a5;
        auto a9 = -a3 + a5;
        auto a10 = -a1 + a3;
        auto a11 = a2 + a4 - a6;
        auto a12 = a2 + a6;
        auto a13 = -a4 - a6;
        auto a14 = -a2 + a4;
        auto m1 = (MFFTELEMRI)c71 * a7;
        auto m2 = (MFFTELEMRI)c72 * a8;
        auto m3 = (MFFTELEMRI)c73 * a9;
        auto m4 = (MFFTELEMRI)c74 * a10;
        auto m5 = -times_pmim<Inverse>((MFFTELEMRI)c75 * a11);
        auto m6 = -times_pmim<Inverse>((MFFTELEMRI)c76 * a12);
        auto m7 = -times_pmim<Inverse>((MFFTELEMRI)c77 * a13);
        auto m8 = -times_pmim<Inverse>((MFFTELEMRI)c78 * a14);
        auto x1 = c0 - m1;
        auto x2 = x1 + m2 + m3;
        auto x3 = x1 - m2 - m4;
        auto x4 = x1 - m3 + m4;
        auto x5 = m5 + m6 - m7;
        auto x6 = m5 - m6 - m8;
        auto x7 = -m5 - m7 - m8;
        Y[bp + stride * (k + 7 * j * m)] = (c0 + a7);
        Y[bp + stride * (k + 7 * j * m + m)] = ((MFFTELEM)w * (x2 - x5));
        Y[bp + stride * (k + 7 * j * m + 2 * m)] = ((MFFTELEM)w2 * (x3 - x6));
        Y[bp + stride * (k + 7 * j * m + 3 * m)] = ((MFFTELEM)w3 * (x4 - x7));
        Y[bp + stride * (k + 7 * j * m + 4 * m)] = ((MFFTELEM)w4 * (x4 + x7));
        Y[bp + stride * (k + 7 * j * m + 5 * m)] = ((MFFTELEM)w5 * (x3 + x6));
        Y[bp + stride * (k + 7 * j * m + 6 * m)] = ((MFFTELEM)w6 * (x2 + x5));
      }
      w = w * w_l;
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

template <bool Inverse>
void fftr8(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  int64_t l = N >> 3;
  int64_t m = 1;
  const double c81 = 0.7071067811865476;   // sqrt(2.0) / 2.0;
  MFFTELEM* tmp;
  const std::complex<double>* __restrict__ W =
      reinterpret_cast<const std::complex<double>*>(COS_SIN_2);
  for (int32_t t = 0; t < e1; t++) {
    auto w = W[0];
    auto w_l = W[3 * (e1 - t) - 1];
    if constexpr (Inverse) {
      w_l = conj(w_l);
    }
    for (int64_t j = 0; j < l; j++) {
      auto w2 = w * w;
      auto w3 = w2 * w;
      auto w4 = w2 * w2;
      auto w5 = w4 * w;
      auto w6 = w3 * w3;
      auto w7 = w4 * w3;
      for (int64_t k = 0; k < m; k++) {
        auto c0 = X[bp + stride * (k + j * m)];
        auto c1 = X[bp + stride * (k + j * m + l * m)];
        auto c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        auto c3 = X[bp + stride * (k + j * m + 3 * l * m)];
        auto c4 = X[bp + stride * (k + j * m + 4 * l * m)];
        auto c5 = X[bp + stride * (k + j * m + 5 * l * m)];
        auto c6 = X[bp + stride * (k + j * m + 6 * l * m)];
        auto c7 = X[bp + stride * (k + j * m + 7 * l * m)];
        auto d0 = c0 + c4;
        auto d1 = c0 - c4;
        auto d2 = c2 + c6;
        auto d3 = times_pmim<Inverse>((c2 - c6));
        auto d4 = c1 + c5;
        auto d5 = c1 - c5;
        auto d6 = c3 + c7;
        auto d7 = c3 - c7;
        auto m0 = d0 + d2;
        auto m1 = d0 - d2;
        auto m2 = d4 + d6;
        auto m3 = times_pmim<Inverse>((d4 - d6));
        auto m4 = (MFFTELEMRI)c81 * (d5 - d7);
        auto m5 = times_pmim<Inverse>((MFFTELEMRI)c81 * (d5 + d7));
        auto m6 = d1 + m4;
        auto m7 = d1 - m4;
        auto m8 = d3 + m5;
        auto m9 = d3 - m5;
        Y[bp + stride * (k + 8 * j * m)] = (m0 + m2);
        Y[bp + stride * (k + 8 * j * m + m)] = ((MFFTELEM)w * (m6 + m8));
        Y[bp + stride * (k + 8 * j * m + 2 * m)] = ((MFFTELEM)w2 * (m1 + m3));
        Y[bp + stride * (k + 8 * j * m + 3 * m)] = ((MFFTELEM)w3 * (m7 - m9));
        Y[bp + stride * (k + 8 * j * m + 4 * m)] = ((MFFTELEM)w4 * (m0 - m2));
        Y[bp + stride * (k + 8 * j * m + 5 * m)] = ((MFFTELEM)w5 * (m7 + m9));
        Y[bp + stride * (k + 8 * j * m + 6 * m)] = ((MFFTELEM)w6 * (m1 - m3));
        Y[bp + stride * (k + 8 * j * m + 7 * m)] = ((MFFTELEM)w7 * (m6 - m8));
      }
      w = w * w_l;
    }
    l /= 8;
    m *= 8;
    tmp = X;
    X = Y;
    Y = tmp;
  }
  *XX = Y;
  *YY = X;
}

template <bool Inverse>
void fftr9(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  int64_t l = N / 9;
  int64_t m = 1;
  const double c90 = 0.5;
  const double c91 = 3.0 / 2.0;
  const double c93 = 0.766044443118978;    // (2 * cos(u) - cos(2 * u) - cos(4 * u)) / 3.0;
  const double c94 = 0.9396926207859083;   // (cos(u) + cos(2 * u) - 2 * cos(4 * u)) / 3.0;
  const double c95 = -0.1736481776669304;  // (cos(u) - 2 * cos(2 * u) + cos(4 * u)) / 3.0;
  const double su = 0.6427876096865393;    // sin(u);
  const double s2u = 0.984807753012208;    // sin(2 * u);
  const double s3u = 0.8660254037844387;   // sin(3 * u);
  const double s4u = 0.3420201433256689;   // sin(4 * u);
  MFFTELEM* tmp;
  const std::complex<double>* __restrict__ W =
      reinterpret_cast<const std::complex<double>*>(COS_SIN_3);
  for (int32_t t = 0; t < e1; t++) {
    auto w = W[0];
    auto w_l = W[2 * (e1 - t) - 1];
    if constexpr (Inverse) {
      w_l = conj(w_l);
    }
    for (int64_t j = 0; j < l; j++) {
      auto w2 = w * w;
      auto w3 = w2 * w;
      auto w4 = w2 * w2;
      auto w5 = w4 * w;
      auto w6 = w3 * w3;
      auto w7 = w4 * w3;
      auto w8 = w4 * w4;
      for (int64_t k = 0; k < m; k++) {
        auto c0 = X[bp + stride * (k + j * m)];
        auto c1 = X[bp + stride * (k + j * m + l * m)];
        auto c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        auto c3 = X[bp + stride * (k + j * m + 3 * l * m)];
        auto c4 = X[bp + stride * (k + j * m + 4 * l * m)];
        auto c5 = X[bp + stride * (k + j * m + 5 * l * m)];
        auto c6 = X[bp + stride * (k + j * m + 6 * l * m)];
        auto c7 = X[bp + stride * (k + j * m + 7 * l * m)];
        auto c8 = X[bp + stride * (k + j * m + 8 * l * m)];
        auto t1 = c1 + c8;
        auto t2 = c2 + c7;
        auto t3 = c3 + c6;
        auto t4 = c4 + c5;
        auto t5 = t1 + t2 + t4;
        auto t6 = c1 - c8;
        auto t7 = c7 - c2;
        auto t8 = c3 - c6;
        auto t9 = c4 - c5;
        auto t10 = t6 + t7 + t9;
        auto t11 = t1 - t2;
        auto t12 = t2 - t4;
        auto t13 = t7 - t6;
        auto t14 = t7 - t9;
        auto m0 = c0 + t3 + t5;
        auto m1 = (MFFTELEMRI)c91 * t3;
        auto m2 = -t5 * (MFFTELEMRI)c90;
        auto t15 = -t12 - t11;
        auto m3 = (MFFTELEMRI)c93 * t11;
        auto m4 = (MFFTELEMRI)c94 * t12;
        auto m5 = (MFFTELEMRI)c95 * t15;
        auto s0 = -m3 - m4;
        auto s1 = m5 - m4;
        auto m6 = times_pmim<Inverse>((MFFTELEMRI)s3u * t10);
        auto m7 = times_pmim<Inverse>((MFFTELEMRI)s3u * t8);
        auto t16 = -t13 + t14;
        auto m8 = -times_pmim<Inverse>((MFFTELEMRI)su * t13);
        auto m9 = -times_pmim<Inverse>((MFFTELEMRI)s4u * t14);
        auto m10 = -times_pmim<Inverse>((MFFTELEMRI)s2u * t16);
        auto s2 = -m8 - m9;
        auto s3 = m9 - m10;
        auto s4 = m0 + m2 + m2;
        auto s5 = s4 - m1;
        auto s6 = s4 + m2;
        auto s7 = s5 - s0;
        auto s8 = s1 + s5;
        auto s9 = s0 - s1 + s5;
        auto s10 = m7 - s2;
        auto s11 = m7 - s3;
        auto s12 = m7 + s2 + s3;
        Y[bp + stride * (k + 9 * j * m)] = m0;
        Y[bp + stride * (k + 9 * j * m + m)] = ((MFFTELEM)w * (s7 + s10));
        Y[bp + stride * (k + 9 * j * m + 2 * m)] = ((MFFTELEM)w2 * (s8 - s11));
        Y[bp + stride * (k + 9 * j * m + 3 * m)] = ((MFFTELEM)w3 * (s6 + m6));
        Y[bp + stride * (k + 9 * j * m + 4 * m)] = ((MFFTELEM)w4 * (s9 + s12));
        Y[bp + stride * (k + 9 * j * m + 5 * m)] = ((MFFTELEM)w5 * (s9 - s12));
        Y[bp + stride * (k + 9 * j * m + 6 * m)] = ((MFFTELEM)w6 * (s6 - m6));
        Y[bp + stride * (k + 9 * j * m + 7 * m)] = ((MFFTELEM)w7 * (s8 + s11));
        Y[bp + stride * (k + 9 * j * m + 8 * m)] = ((MFFTELEM)w8 * (s7 - s10));
      }
      w = w * w_l;
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

template void fftr2<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr2<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr3<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr3<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr4<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr4<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr5<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr5<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr7<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr7<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr8<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr8<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr9<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
template void fftr9<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
           const int64_t stride, int32_t flags);
