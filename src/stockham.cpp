
#include <complex>
#include <cstdint>

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"
#include "weights.hpp"

void fftr2(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 2;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  std::complex<double> w, w_l;
  MFFTELEM c0, c1;
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_2);
  for (int32_t t = 0; t < e1; t++) {
    w = W[0];
    w_l = inverse ? conj(W[e1 - t - 1]) : W[e1 - t - 1];
    for (int64_t j = 0; j < l; j++) {
      for (int64_t k = 0; k < m; k++) {
        c0 = X[bp + stride * (k + j * m)];
        c1 = X[bp + stride * (k + j * m + l * m)];
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

void fftr3(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 3;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c30 = 0.5;
  const double c31 = 0.8660254037844386;  // sin(M_PI / 3.0);
  std::complex<double> w, w_l, w2;
  MFFTELEM c0, c1, c2, d0, d1, d2;
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_3);
  for (int32_t t = 0; t < e1; t++) {
    w = W[0];
    w_l = inverse ? conj(W[e1 - t - 1]) : W[e1 - t - 1];
    for (int64_t j = 0; j < l; j++) {
      for (int64_t k = 0; k < m; k++) {
        c0 = X[bp + stride * (k + j * m)];
        c1 = X[bp + stride * (k + j * m + l * m)];
        c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        d0 = c1 + c2;
        d1 = c0 - (MFFTELEM)c30 * d0;
        d2 = times_pmim((MFFTELEM)c31 * (c1 - c2), inverse);
        w2 = w * w;
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

void fftr4(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N >> 2;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  std::complex<double> w, w_l, w2, w3;
  MFFTELEM c0, c1, c2, c3, d0, d1, d2, d3;
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_2);
  for (int32_t t = 0; t < e1; t++) {
    w = W[0];
    w_l = inverse ? conj(W[2 * (e1 - t) - 1]) : W[2 * (e1 - t) - 1];
    for (int64_t j = 0; j < l; j++) {
      w2 = w * w;
      w3 = w2 * w;
      for (int64_t k = 0; k < m; k++) {
        c0 = X[bp + stride * (k + j * m)];
        c1 = X[bp + stride * (k + j * m + m * l)];
        c2 = X[bp + stride * (k + j * m + 2 * m * l)];
        c3 = X[bp + stride * (k + j * m + 3 * m * l)];
        d0 = c0 + c2;
        d1 = c0 - c2;
        d2 = c1 + c3;
        d3 = times_pmim(c1 - c3, inverse);
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

void fftr5(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 5;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c50 = 0.25;
  const double c51 = 0.9510565162951535;  // sin(2.0 * M_PI / 5.0);
  const double c52 = 0.5590169943749475;  // sqrt(5.0) / 4.0;
  const double c53 =
      0.6180339887498949;  // sin(M_PI / 5.0) / sin(2.0 * M_PI / 5.0);
  std::complex<double> w, w_l, w2, w3, w4;
  MFFTELEM c0, c1, c2, c3, c4, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10;
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_5);
  for (int32_t t = 0; t < e1; t++) {
    w = W[0];
    w_l = inverse ? conj(W[e1 - t - 1]) : W[e1 - t - 1];
    for (int64_t j = 0; j < l; j++) {
      w2 = w * w;
      w3 = w2 * w;
      w4 = w2 * w2;
      for (int64_t k = 0; k < m; k++) {
        c0 = X[bp + stride * (k + j * m)];
        c1 = X[bp + stride * (k + j * m + l * m)];
        c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        c3 = X[bp + stride * (k + j * m + 3 * l * m)];
        c4 = X[bp + stride * (k + j * m + 4 * l * m)];
        d0 = c1 + c4;
        d1 = c2 + c3;
        d2 = (MFFTELEM)c51 * (c1 - c4);
        d3 = (MFFTELEM)c51 * (c2 - c3);
        d4 = d0 + d1;
        d5 = (MFFTELEM)c52 * (d0 - d1);
        d6 = c0 - (MFFTELEM)c50 * d4;
        d7 = d6 + d5;
        d8 = d6 - d5;
        d9 = times_pmim(d2 + (MFFTELEM)c53 * d3, inverse);
        d10 = times_pmim((MFFTELEM)c53 * d2 - d3, inverse);
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

void fftr7(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 7;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c71 =
      0.1666666666666666;  // -(cos(u) + cos(2 * u) + cos(3 * u)) / 3.0;
  const double c72 =
      0.7901564685254002;  // (2 * cos(u) - cos(2 * u) - cos(3 * u)) / 3.0;
  const double c73 =
      0.05585426728964774;  // (cos(u) - 2 * cos(2 * u) + cos(3 * u)) / 3.0;
  const double c74 =
      0.7343022012357524;  // (cos(u) + cos(2 * u) - 2 * cos(3 * u)) / 3.0;
  const double c75 =
      0.4409585518440984;  // (sin(u) + sin(2 * u) - sin(3 * u)) / 3.0;
  const double c76 =
      0.34087293062393137;  // (2 * sin(u) - sin(2 * u) + sin(3 * u)) / 3.0;
  const double c77 =
      0.5339693603377252;  // (-sin(u) + 2 * sin(2 * u) + sin(3 * u)) / 3.0;
  const double c78 =
      0.8748422909616567;  // (sin(u) + sin(2 * u) + 2 * sin(3 * u)) / 3.0;
  std::complex<double> w, w_l, w2, w3, w4, w5, w6;
  MFFTELEM c0, c1, c2, c3, c4, c5, c6;
  MFFTELEM a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14;
  MFFTELEM m1, m2, m3, m4, m5, m6, m7, m8;
  MFFTELEM x1, x2, x3, x4, x5, x6, x7;
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_7);
  for (int32_t t = 0; t < e1; t++) {
    w = W[0];
    w_l = inverse ? conj(W[e1 - t - 1]) : W[e1 - t - 1];
    for (int64_t j = 0; j < l; j++) {
      w2 = w * w;
      w3 = w2 * w;
      w4 = w2 * w2;
      w5 = w4 * w;
      w6 = w3 * w3;
      for (int64_t k = 0; k < m; k++) {
        c0 = X[bp + stride * (k + j * m)];
        c1 = X[bp + stride * (k + j * m + l * m)];
        c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        c3 = X[bp + stride * (k + j * m + 3 * l * m)];
        c4 = X[bp + stride * (k + j * m + 4 * l * m)];
        c5 = X[bp + stride * (k + j * m + 5 * l * m)];
        c6 = X[bp + stride * (k + j * m + 6 * l * m)];
        a1 = c1 + c6;
        a2 = c1 - c6;
        a3 = c2 + c5;
        a4 = c2 - c5;
        a5 = c3 + c4;
        a6 = c3 - c4;
        a7 = a1 + a3 + a5;
        a8 = a1 - a5;
        a9 = -a3 + a5;
        a10 = -a1 + a3;
        a11 = a2 + a4 - a6;
        a12 = a2 + a6;
        a13 = -a4 - a6;
        a14 = -a2 + a4;
        m1 = (MFFTELEM)c71 * a7;
        m2 = (MFFTELEM)c72 * a8;
        m3 = (MFFTELEM)c73 * a9;
        m4 = (MFFTELEM)c74 * a10;
        m5 = -times_pmim((MFFTELEM)c75 * a11, inverse);
        m6 = -times_pmim((MFFTELEM)c76 * a12, inverse);
        m7 = -times_pmim((MFFTELEM)c77 * a13, inverse);
        m8 = -times_pmim((MFFTELEM)c78 * a14, inverse);
        x1 = c0 - m1;
        x2 = x1 + m2 + m3;
        x3 = x1 - m2 - m4;
        x4 = x1 - m3 + m4;
        x5 = m5 + m6 - m7;
        x6 = m5 - m6 - m8;
        x7 = -m5 - m7 - m8;
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

void fftr8(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N >> 3;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c81 = 0.7071067811865476;   // sqrt(2.0) / 2.0;
  const double c82 = -0.7071067811865476;  // -sqrt(2.0) / 2.0;
  std::complex<double> w, w_l, w2, w3, w4, w5, w6, w7;
  MFFTELEM c0, c1, c2, c3, c4, c5, c6, c7;
  MFFTELEM d0, d1, d2, d3, d4, d5, d6, d7;
  MFFTELEM m0, m1, m2, m3, m4, m5, m6, m7, m8, m9;
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_2);
  for (int32_t t = 0; t < e1; t++) {
    w = W[0];
    w_l = inverse ? conj(W[3 * (e1 - t) - 1]) : W[3 * (e1 - t) - 1];
    for (int64_t j = 0; j < l; j++) {
      w2 = w * w;
      w3 = w2 * w;
      w4 = w2 * w2;
      w5 = w4 * w;
      w6 = w3 * w3;
      w7 = w4 * w3;
      for (int64_t k = 0; k < m; k++) {
        c0 = X[bp + stride * (k + j * m)];
        c1 = X[bp + stride * (k + j * m + l * m)];
        c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        c3 = X[bp + stride * (k + j * m + 3 * l * m)];
        c4 = X[bp + stride * (k + j * m + 4 * l * m)];
        c5 = X[bp + stride * (k + j * m + 5 * l * m)];
        c6 = X[bp + stride * (k + j * m + 6 * l * m)];
        c7 = X[bp + stride * (k + j * m + 7 * l * m)];
        d0 = c0 + c4;
        d1 = c0 - c4;
        d2 = c2 + c6;
        d3 = times_pmim((c2 - c6), inverse);
        d4 = c1 + c5;
        d5 = c1 - c5;
        d6 = c3 + c7;
        d7 = c3 - c7;
        m0 = d0 + d2;
        m1 = d0 - d2;
        m2 = d4 + d6;
        m3 = times_pmim((d4 - d6), inverse);
        m4 = (MFFTELEM)c81 * (d5 - d7);
        m5 = -times_pmim((MFFTELEM)c82 * (d5 + d7), inverse);
        m6 = d1 + m4;
        m7 = d1 - m4;
        m8 = d3 + m5;
        m9 = d3 - m5;
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

void fftr9(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  int64_t l = N / 9;
  int64_t m = 1;
  const bool inverse = (flags & P_INVERSE);
  const double c90 = 0.5;
  const double c91 = 3.0 / 2.0;
  const double c93 =
      0.766044443118978;  // (2 * cos(u) - cos(2 * u) - cos(4 * u)) / 3.0;
  const double c94 =
      0.9396926207859083;  // (cos(u) + cos(2 * u) - 2 * cos(4 * u)) / 3.0;
  const double c95 =
      -0.1736481776669304;  // (cos(u) - 2 * cos(2 * u) + cos(4 * u)) / 3.0;
  const double su = 0.6427876096865393;   // sin(u);
  const double s2u = 0.984807753012208;   // sin(2 * u);
  const double s3u = 0.8660254037844387;  // sin(3 * u);
  const double s4u = 0.3420201433256689;  // sin(4 * u);
  std::complex<double> w, w_l, w2, w3, w4, w5, w6, w7, w8;
  MFFTELEM c0, c1, c2, c3, c4, c5, c6, c7, c8;
  MFFTELEM t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
      t16;
  MFFTELEM m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10;
  MFFTELEM s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12;
  MFFTELEM *tmp;
  const std::complex<double> *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(COS_SIN_3);
  for (int32_t t = 0; t < e1; t++) {
    w = W[0];
    w_l = inverse ? conj(W[2 * (e1 - t) - 1]) : W[2 * (e1 - t) - 1];
    for (int64_t j = 0; j < l; j++) {
      w2 = w * w;
      w3 = w2 * w;
      w4 = w2 * w2;
      w5 = w4 * w;
      w6 = w3 * w3;
      w7 = w4 * w3;
      w8 = w4 * w4;
      for (int64_t k = 0; k < m; k++) {
        c0 = X[bp + stride * (k + j * m)];
        c1 = X[bp + stride * (k + j * m + l * m)];
        c2 = X[bp + stride * (k + j * m + 2 * l * m)];
        c3 = X[bp + stride * (k + j * m + 3 * l * m)];
        c4 = X[bp + stride * (k + j * m + 4 * l * m)];
        c5 = X[bp + stride * (k + j * m + 5 * l * m)];
        c6 = X[bp + stride * (k + j * m + 6 * l * m)];
        c7 = X[bp + stride * (k + j * m + 7 * l * m)];
        c8 = X[bp + stride * (k + j * m + 8 * l * m)];
        t1 = c1 + c8;
        t2 = c2 + c7;
        t3 = c3 + c6;
        t4 = c4 + c5;
        t5 = t1 + t2 + t4;
        t6 = c1 - c8;
        t7 = c7 - c2;
        t8 = c3 - c6;
        t9 = c4 - c5;
        t10 = t6 + t7 + t9;
        t11 = t1 - t2;
        t12 = t2 - t4;
        t13 = t7 - t6;
        t14 = t7 - t9;
        m0 = c0 + t3 + t5;
        m1 = (MFFTELEM)c91 * t3;
        m2 = -t5 * (MFFTELEM)c90;
        t15 = -t12 - t11;
        m3 = (MFFTELEM)c93 * t11;
        m4 = (MFFTELEM)c94 * t12;
        m5 = (MFFTELEM)c95 * t15;
        s0 = -m3 - m4;
        s1 = m5 - m4;
        m6 = times_pmim((MFFTELEM)s3u * t10, inverse);
        m7 = times_pmim((MFFTELEM)s3u * t8, inverse);
        t16 = -t13 + t14;
        m8 = -times_pmim((MFFTELEM)su * t13, inverse);
        m9 = -times_pmim((MFFTELEM)s4u * t14, inverse);
        m10 = -times_pmim((MFFTELEM)s2u * t16, inverse);
        s2 = -m8 - m9;
        s3 = m9 - m10;
        s4 = m0 + m2 + m2;
        s5 = s4 - m1;
        s6 = s4 + m2;
        s7 = s5 - s0;
        s8 = s1 + s5;
        s9 = s0 - s1 + s5;
        s10 = m7 - s2;
        s11 = m7 - s3;
        s12 = m7 + s2 + s3;
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