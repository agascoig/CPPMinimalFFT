
// small.cpp - PFA for small block sizes ( N<= 28).

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"

static inline void butterfly_3_1(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                 const int8_t *n_map, const int64_t N1, const int64_t N2,
                                 const int64_t bp, const int64_t stride, const int32_t flags)
{
  const bool inverse = (flags & P_INVERSE);
  const MFFTELEMRI c30 = 0.5;
  const MFFTELEMRI c31 = 0.8660254037844386; // sin(M_PI / 3.0);
  for (int8_t i = 0; i < N2; ++i)
  {
    const int8_t iN1 = i * N1;
    auto c0 = X[bp + stride * n_map[iN1]];
    auto c1 = X[bp + stride * n_map[iN1 + 1]];
    auto c2 = X[bp + stride * n_map[iN1 + 2]];
    auto d0 = c1 + c2;
    auto d1 = c0 - c30 * d0;
    auto d2 = times_pmim(c31 * (c1 - c2), inverse);
    Y[bp + stride * (iN1)] = c0 + d0;
    Y[bp + stride * (iN1 + 1)] = d1 + d2;
    Y[bp + stride * (iN1 + 2)] = d1 - d2;
  }
}

static inline void butterfly_4_1(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                 const int8_t *n_map, const int64_t N1, const int64_t N2,
                                 const int64_t bp, const int64_t stride, const int32_t flags)
{
  const bool inverse = (flags & P_INVERSE);
  for (int8_t i = 0; i < N2; ++i)
  {
    const int8_t iN1 = i * N1;
    auto c0 = X[bp + stride * n_map[iN1]];
    auto c1 = X[bp + stride * n_map[iN1 + 1]];
    auto c2 = X[bp + stride * n_map[iN1 + 2]];
    auto c3 = X[bp + stride * n_map[iN1 + 3]];
    auto d0 = c0 + c2;
    auto d1 = c0 - c2;
    auto d2 = c1 + c3;
    auto d3 = times_pmim(c1 - c3, inverse);
    Y[bp + stride * (iN1)] = d0 + d2;
    Y[bp + stride * (iN1 + 1)] = d1 + d3;
    Y[bp + stride * (iN1 + 2)] = d0 - d2;
    Y[bp + stride * (iN1 + 3)] = d1 - d3;
  }
}

static inline void butterfly_5_1(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                 const int8_t *n_map, const int64_t N1, const int64_t N2,
                                 const int64_t bp, const int64_t stride, const int32_t flags)
{
  const bool inverse = (flags & P_INVERSE);
  const MFFTELEMRI c50 = 0.25;
  const MFFTELEMRI c51 = 0.9510565162951535; // sin(2.0 * M_PI / 5.0);
  const MFFTELEMRI c52 = 0.5590169943749475; // sqrt(5.0) / 4.0;
  const MFFTELEMRI c53 = 0.6180339887498949; // sin(M_PI / 5.0) / sin(2.0 * M_PI / 5.0);
  for (int8_t i = 0; i < N2; ++i)
  {
    const int8_t iN1 = i * N1;
    auto c0 = X[bp + stride * n_map[iN1]];
    auto c1 = X[bp + stride * n_map[iN1 + 1]];
    auto c2 = X[bp + stride * n_map[iN1 + 2]];
    auto c3 = X[bp + stride * n_map[iN1 + 3]];
    auto c4 = X[bp + stride * n_map[iN1 + 4]];
    auto d0 = c1 + c4;
    auto d1 = c2 + c3;
    auto d2 = c51 * (c1 - c4);
    auto d3 = c51 * (c2 - c3);
    auto d4 = d0 + d1;
    auto d5 = c52 * (d0 - d1);
    auto d6 = c0 - c50 * d4;
    auto d7 = d6 + d5;
    auto d8 = d6 - d5;
    auto d9 = times_pmim(d2 + c53 * d3, inverse);
    auto d10 = times_pmim(c53 * d2 - d3, inverse);
    Y[bp + stride * (iN1)] = (c0 + d4);
    Y[bp + stride * (iN1 + 1)] = (d7 + d9);
    Y[bp + stride * (iN1 + 2)] = (d8 + d10);
    Y[bp + stride * (iN1 + 3)] = (d8 - d10);
    Y[bp + stride * (iN1 + 4)] = (d7 - d9);
  }
}

static inline void butterfly_7_1(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                 const int8_t *n_map, const int64_t N1, const int64_t N2,
                                 const int64_t bp, const int64_t stride, const int32_t flags)
{
  const bool inverse = (flags & P_INVERSE);
  const MFFTELEMRI c71 = 0.1666666666666666;   // -(cos(u) + cos(2 * u) + cos(3 * u)) / 3.0;
  const MFFTELEMRI c72 = 0.7901564685254002;   // (2 * cos(u) - cos(2 * u) - cos(3 * u)) / 3.0;
  const MFFTELEMRI c73 = 0.05585426728964774;  // (cos(u) - 2 * cos(2 * u) + cos(3 * u)) / 3.0;
  const MFFTELEMRI c74 = 0.7343022012357524;   // (cos(u) + cos(2 * u) - 2 * cos(3 * u)) / 3.0;
  const MFFTELEMRI c75 = -0.4409585518440984;  // (sin(u) + sin(2 * u) - sin(3 * u)) / 3.0;
  const MFFTELEMRI c76 = -0.34087293062393137; // (2 * sin(u) - sin(2 * u) + sin(3 * u)) / 3.0;
  const MFFTELEMRI c77 = -0.5339693603377252;  // (-sin(u) + 2 * sin(2 * u) + sin(3 * u)) / 3.0;
  const MFFTELEMRI c78 = -0.8748422909616567;  // (sin(u) + sin(2 * u) + 2 * sin(3 * u)) / 3.0;
  for (int i = 0; i < N2; ++i)
  {
    const int8_t iN1 = i * N1;
    auto c0 = X[bp + stride * n_map[iN1]];
    auto c1 = X[bp + stride * n_map[iN1 + 1]];
    auto c2 = X[bp + stride * n_map[iN1 + 2]];
    auto c3 = X[bp + stride * n_map[iN1 + 3]];
    auto c4 = X[bp + stride * n_map[iN1 + 4]];
    auto c5 = X[bp + stride * n_map[iN1 + 5]];
    auto c6 = X[bp + stride * n_map[iN1 + 6]];
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
    auto m1 = c71 * a7;
    auto m2 = c72 * a8;
    auto m3 = c73 * a9;
    auto m4 = c74 * a10;
    auto m5 = times_pmim(c75 * a11, inverse);
    auto m6 = times_pmim(c76 * a12, inverse);
    auto m7 = times_pmim(c77 * a13, inverse);
    auto m8 = times_pmim(c78 * a14, inverse);
    auto x1 = c0 - m1;
    auto x2 = x1 + m2 + m3;
    auto x3 = x1 - m2 - m4;
    auto x4 = x1 - m3 + m4;
    auto x5 = m5 + m6 - m7;
    auto x6 = m5 - m6 - m8;
    auto x7 = -m5 - m7 - m8;
    Y[bp + stride * (iN1)] = (c0 + a7);
    Y[bp + stride * (iN1 + 1)] = (x2 - x5);
    Y[bp + stride * (iN1 + 2)] = (x3 - x6);
    Y[bp + stride * (iN1 + 3)] = (x4 - x7);
    Y[bp + stride * (iN1 + 4)] = (x4 + x7);
    Y[bp + stride * (iN1 + 5)] = (x3 + x6);
    Y[bp + stride * (iN1 + 6)] = (x2 + x5);
  }
}

void butterfly_8_1(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                   const int8_t *n_map, const int64_t N1, const int64_t N2,
                   const int64_t bp, const int64_t stride, const int32_t flags)
{
  const bool inverse = (flags & P_INVERSE);
  const double c81 = 0.7071067811865476; // sqrt(2.0) / 2.0;
  for (int8_t i = 0; i < N2; ++i)
  {
    const int8_t iN1 = i * N1;
    auto c0 = X[bp + stride * n_map[iN1]];
    auto c1 = X[bp + stride * n_map[iN1 + 1]];
    auto c2 = X[bp + stride * n_map[iN1 + 2]];
    auto c3 = X[bp + stride * n_map[iN1 + 3]];
    auto c4 = X[bp + stride * n_map[iN1 + 4]];
    auto c5 = X[bp + stride * n_map[iN1 + 5]];
    auto c6 = X[bp + stride * n_map[iN1 + 6]];
    auto c7 = X[bp + stride * n_map[iN1 + 7]];
    auto d0 = c0 + c4;
    auto d1 = c0 - c4;
    auto d2 = c2 + c6;
    auto d3 = times_pmim(c2 - c6, inverse);
    auto d4 = c1 + c5;
    auto d5 = c1 - c5;
    auto d6 = c3 + c7;
    auto d7 = c3 - c7;
    auto m0 = d0 + d2;
    auto m1 = d0 - d2;
    auto m2 = d4 + d6;
    auto m3 = times_pmim((d4 - d6), inverse);
    auto m4 = (MFFTELEMRI)c81 * (d5 - d7);
    auto m5 = times_pmim((MFFTELEMRI)c81 * (d5 + d7), inverse);
    auto m6 = d1 + m4;
    auto m7 = d1 - m4;
    auto m8 = d3 + m5;
    auto m9 = d3 - m5;
    Y[bp + stride * (iN1)] = (m0 + m2);
    Y[bp + stride * (iN1 + 1)] = (m6 + m8);
    Y[bp + stride * (iN1 + 2)] = (m1 + m3);
    Y[bp + stride * (iN1 + 3)] = (m7 - m9);
    Y[bp + stride * (iN1 + 4)] = (m0 - m2);
    Y[bp + stride * (iN1 + 5)] = (m7 + m9);
    Y[bp + stride * (iN1 + 6)] = (m1 - m3);
    Y[bp + stride * (iN1 + 7)] = (m6 - m8);
  }
}

static void butterfly_9_1(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                          const int8_t *n_map, const int64_t N1, const int64_t N2,
                          const int64_t bp, const int64_t stride, const int32_t flags)
{
  const bool inverse = (flags & P_INVERSE);
  const double c90 = 0.5;
  const double c91 = 3.0 / 2.0;
  const double c93 = 0.766044443118978;   // (2 * cos(u) - cos(2 * u) - cos(4 * u)) / 3.0;
  const double c94 = 0.9396926207859083;  // (cos(u) + cos(2 * u) - 2 * cos(4 * u)) / 3.0;
  const double c95 = -0.1736481776669304; // (cos(u) - 2 * cos(2 * u) + cos(4 * u)) / 3.0;
  const double su = 0.6427876096865393;   // sin(u);
  const double s2u = 0.984807753012208;   // sin(2 * u);
  const double s3u = 0.8660254037844387;  // sin(3 * u);
  const double s4u = 0.3420201433256689;  // sin(4 * u);
  for (int i = 0; i < N2; ++i)
  {
    const int8_t iN1 = i * N1;
        auto c0 = X[bp + stride * n_map[iN1]];
        auto c1 = X[bp + stride * n_map[iN1+1]];
        auto c2 = X[bp + stride * n_map[iN1+2]];
        auto c3 = X[bp + stride * n_map[iN1+3]];
        auto c4 = X[bp + stride * n_map[iN1+4]];
        auto c5 = X[bp + stride * n_map[iN1+5]];
        auto c6 = X[bp + stride * n_map[iN1+6]];
        auto c7 = X[bp + stride * n_map[iN1+7]];
        auto c8 = X[bp + stride * n_map[iN1+8]];
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
        auto m6 = times_pmim((MFFTELEMRI)s3u * t10, inverse);
        auto m7 = times_pmim((MFFTELEMRI)s3u * t8, inverse);
        auto t16 = -t13 + t14;
        auto m8 = -times_pmim((MFFTELEMRI)su * t13, inverse);
        auto m9 = -times_pmim((MFFTELEMRI)s4u * t14, inverse);
        auto m10 = -times_pmim((MFFTELEMRI)s2u * t16, inverse);
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
        Y[bp + stride * (iN1)] = m0;
        Y[bp + stride * (iN1+1)] = (s7 + s10);
        Y[bp + stride * (iN1+2)] = (s8 - s11);
        Y[bp + stride * (iN1+3)] = (s6 + m6);
        Y[bp + stride * (iN1+4)] = (s9 + s12);
        Y[bp + stride * (iN1+5)] = (s9 - s12);
        Y[bp + stride * (iN1+6)] = (s6 - m6);
        Y[bp + stride * (iN1+7)] = (s8 + s11);
        Y[bp + stride * (iN1+8)] = (s7 - s10);
  }
}

static inline void butterfly_2_2(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                 const int8_t *k_map, const int64_t N1, const int64_t N2,
                                 const int64_t bp, const int64_t stride, const int32_t flags)
{
  for (int8_t i = 0; i < N1; ++i)
  {
    auto c0 = X[bp + stride * (i)];
    auto c1 = X[bp + stride * (i + N1)];
    Y[bp + stride * k_map[(i)]] = (c0 + c1);
    Y[bp + stride * k_map[(i + N1)]] = (c0 - c1);
  }
}

static inline void butterfly_3_2(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                 const int8_t *k_map, const int64_t N1, const int64_t N2,
                                 const int64_t bp, const int64_t stride, const int32_t flags)
{
  const bool inverse = (flags & P_INVERSE);
  const MFFTELEMRI c30 = 0.5;
  const MFFTELEMRI c31 = 0.8660254037844386; // sin(M_PI / 3.0);
  for (int8_t i = 0; i < N1; ++i)
  {
    auto c0 = X[bp + stride * (i)];
    auto c1 = X[bp + stride * (i + N1)];
    auto c2 = X[bp + stride * (i + 2 * N1)];
    auto d0 = c1 + c2;
    auto d1 = c0 - c30 * d0;
    auto d2 = times_pmim(c31 * (c1 - c2), inverse);
    Y[bp + stride * k_map[(i)]] = (c0 + d0);
    Y[bp + stride * k_map[(i + N1)]] = (d1 + d2);
    Y[bp + stride * k_map[(i + 2 * N1)]] = (d1 - d2);
  }
}

static inline void butterfly_4_2(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                                 const int8_t *k_map, const int64_t N1, const int64_t N2,
                                 const int64_t bp, const int64_t stride, const int32_t flags)
{
  const bool inverse = (flags & P_INVERSE);
  for (int8_t i = 0; i < N1; ++i)
  {
    auto c0 = X[bp + stride * i];
    auto c1 = X[bp + stride * (i+N1)];
    auto c2 = X[bp + stride * (i+2*N1)];
    auto c3 = X[bp + stride * (i+3*N1)];
    auto d0 = c0 + c2;
    auto d1 = c0 - c2;
    auto d2 = c1 + c3;
    auto d3 = times_pmim(c1 - c3, inverse);
    Y[bp + stride * k_map[i]] = d0 + d2;
    Y[bp + stride * k_map[i + N1]] = d1 + d3;
    Y[bp + stride * k_map[i + 2*N1]] = d0 - d2;
    Y[bp + stride * k_map[i + 3*N1]] = d1 - d3;
  }
}

void small_1(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  Y[bp] = X[bp];
}

void small_2(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  Y[bp] = X[bp] + X[bp + stride];
  Y[bp + stride] = X[bp] - X[bp + stride];
}

void small_3(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 1, 2};
  butterfly_3_1(Y, X, n_map, 3, 1, bp, stride, flags);
}

void small_4(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 1, 2, 3};
  butterfly_4_1(Y, X, n_map, 4, 1, bp, stride, flags);
}

void small_5(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  const bool inverse = (flags & P_INVERSE);
  const MFFTELEMRI c50 = 0.25;
  const MFFTELEMRI c51 = 0.9510565162951535; // sin(2.0 * M_PI / 5.0);
  const MFFTELEMRI c52 = 0.5590169943749475; // sqrt(5.0) / 4.0;
  const MFFTELEMRI c53 = 0.6180339887498949; // sin(M_PI / 5.0) / sin(2.0 * M_PI / 5.0);
  auto c0 = X[bp];
  auto c1 = X[bp + stride];
  auto c2 = X[bp + stride * 2];
  auto c3 = X[bp + stride * 3];
  auto c4 = X[bp + stride * 4];
  auto d0 = c1 + c4;
  auto d1 = c2 + c3;
  auto d2 = c51 * (c1 - c4);
  auto d3 = c51 * (c2 - c3);
  auto d4 = d0 + d1;
  auto d5 = c52 * (d0 - d1);
  auto d6 = c0 - c50 * d4;
  auto d7 = d6 + d5;
  auto d8 = d6 - d5;
  auto d9 = d2 + c53 * d3;
  d9 = inverse ? std::complex<MFFTELEMRI>(-std::imag(d9), std::real(d9))
               : std::complex<MFFTELEMRI>(std::imag(d9), -std::real(d9));
  auto d10 = c53 * d2 - d3;
  d10 = inverse ? std::complex<MFFTELEMRI>(-std::imag(d10), std::real(d10))
                : std::complex<MFFTELEMRI>(std::imag(d10), -std::real(d10));
  Y[bp] = (c0 + d4);
  Y[bp + stride * 1] = (d7 + d9);
  Y[bp + stride * 2] = (d8 + d10);
  Y[bp + stride * 3] = (d8 - d10);
  Y[bp + stride * 4] = (d7 - d9);
}

void small_6(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  const int8_t n_map[] = {0, 2, 4, 3, 5, 1};
  const int8_t k_map[] = {0, 4, 2, 3, 1, 5};
  butterfly_3_1(Y, X, n_map, 3, 2, bp, stride, flags);
  butterfly_2_2(X, Y, k_map, 3, 2, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_7(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  const int8_t n_map[] = {0, 1, 2, 3, 4, 5, 6};
  butterfly_7_1(Y, X, n_map, 7, 1, bp, stride, flags);
}

void small_8(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  const int8_t n_map[] = {0, 1, 2, 3, 4, 5, 6, 7};
  butterfly_8_1(Y, X, n_map, 8, 1, bp, stride, flags);
}

void small_9(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
             const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  const int8_t n_map[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  butterfly_9_1(Y, X, n_map, 9, 1, bp, stride, flags);
}

void small_10(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  const int8_t n_map[] = {0, 2, 4, 6, 8, 5, 7, 9, 1, 3};
  const int8_t k_map[] = {0, 6, 2, 8, 4, 5, 1, 7, 3, 9};
  butterfly_5_1(Y, X, n_map, 5, 2, bp, stride, flags);
  butterfly_2_2(X, Y, k_map, 5, 2, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_12(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;
  const int8_t n_map[] = {0, 3, 6, 9, 4, 7, 10, 1, 8, 11, 2, 5};
  const int8_t k_map[] = {0, 9, 6, 3, 4, 1, 10, 7, 8, 5, 2, 11};
  butterfly_4_1(Y, X, n_map, 4, 3, bp, stride, flags);
  butterfly_3_2(X, Y, k_map, 4, 3, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_14(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 2, 4, 6, 8, 10, 12, 7, 9, 11, 13, 1, 3, 5};
  const int8_t k_map[] = {0, 8, 2, 10, 4, 12, 6, 7, 1, 9, 3, 11, 5, 13};

  butterfly_7_1(Y, X, n_map, 7, 2, bp, stride, flags);
  butterfly_2_2(X, Y, k_map, 7, 2, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_15(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 3, 6, 9, 12, 10, 13, 1, 4, 7, 5, 8, 11, 14, 2};
  const int8_t k_map[] = {0, 6, 12, 3, 9, 5, 11, 2, 8, 14, 10, 1, 7, 13, 4};

  butterfly_5_1(Y, X, n_map, 5, 3, bp, stride, flags);
  butterfly_3_2(X, Y, k_map, 5, 3, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_18(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  // butterfly 9, 2
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 9, 11, 13, 15, 17, 1, 3, 5, 7};
  const int8_t k_map[] = {0, 10, 2, 12, 4, 14, 6, 16, 8, 9, 1, 11, 3, 13, 5, 15, 7, 17};
  butterfly_9_1(Y, X, n_map, 9, 2, bp, stride, flags);
  butterfly_2_2(X, Y, k_map, 9, 2, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_20(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 4, 8, 12, 16, 5, 9, 13, 17, 1, 10, 14, 18, 2, 6, 15, 19, 3, 7, 11};
  const int8_t k_map[] = {0, 16, 12, 8, 4, 5, 1, 17, 13, 9, 10, 6, 2, 18, 14, 15, 11, 7, 3, 19};
  butterfly_5_1(Y, X, n_map, 5, 4, bp, stride, flags);
  butterfly_4_2(X, Y, k_map, 5, 4, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_21(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 3, 6, 9, 12, 15, 18, 7, 10, 13, 16, 19, 1, 4, 14, 17, 20, 2, 5, 8, 11};
  const int8_t k_map[] = {0, 15, 9, 3, 18, 12, 6, 7, 1, 16, 10, 4, 19, 13, 14, 8, 2, 17, 11, 5, 20};
  butterfly_7_1(Y, X, n_map, 7, 3, bp, stride, flags);
  butterfly_3_2(X, Y, k_map, 7, 3, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_24(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 3, 6, 9, 12, 15, 18, 21, 16, 19, 22, 1, 4, 7, 10, 13, 8, 11, 14, 17, 20, 23, 2, 5};
  const int8_t k_map[] = {0, 9, 18, 3, 12, 21, 6, 15, 8, 17, 2, 11, 20, 5, 14, 23, 16, 1, 10, 19, 4, 13, 22, 7};
  butterfly_8_1(Y, X, n_map, 8, 3, bp, stride, flags);
  butterfly_3_2(X, Y, k_map, 8, 3, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

void small_28(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
              const int64_t stride, const int32_t flags)
{
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int8_t n_map[] = {0, 4, 8, 12, 16, 20, 24, 21, 25, 1, 5, 9, 13, 17, 14, 18, 22, 26, 2, 6, 10, 7, 11, 15, 19, 23, 27, 3};
  const int8_t k_map[] = {0, 8, 16, 24, 4, 12, 20, 7, 15, 23, 3, 11, 19, 27, 14, 22, 2, 10, 18, 26, 6, 21, 1, 9, 17, 25, 5, 13};
  butterfly_7_1(Y, X, n_map, 7, 4, bp, stride, flags);
  butterfly_4_2(X, Y, k_map, 7, 4, bp, stride, flags);
  *YY = X;
  *XX = Y;
}

static const fft_func_t small_funcs[] = {nullptr,
                                         &small_1,
                                         &small_2,
                                         &small_3,
                                         &small_4,
                                         &small_5,
                                         &small_6,
                                         &small_7,
                                         &small_8,
                                         &small_9,
                                         &small_10,
                                         nullptr,
                                         &small_12,
                                         nullptr,
                                         &small_14,
                                         &small_15,
                                         nullptr,
                                         nullptr,
                                         &small_18,
                                         nullptr,
                                         &small_20,
                                         &small_21,
                                         nullptr,
                                         nullptr,
                                         &small_24,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         &small_28};

bool small_available(const int64_t N)
{
  return (N <= SMALL_SZ) && small_funcs[N] != nullptr;
}

void small_dft(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1, const int64_t bp,
               const int64_t stride, const int32_t flags)
{
  minassert(N <= SMALL_SZ, "small_dft limited to N <= SMALL_SZ");
  small_funcs[N](YY, XX, N, e1, bp, stride, flags);
}