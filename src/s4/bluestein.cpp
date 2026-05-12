// Bluestein's FFT algorithm

#include <hwy/highway.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "../CPPMinimalFFT.hpp"
#include "../plan.hpp"

namespace hn = hwy::HWY_NAMESPACE;

#define CCDPTR(x) reinterpret_cast<const double* __restrict__ __attribute__((aligned(ALIGN_SZ)))>(x)
#define CDPTR(x) reinterpret_cast<double* __restrict__ __attribute__((aligned(ALIGN_SZ)))>(x)
#define CCFPTR(x) reinterpret_cast<const float* __restrict__>(__builtin_assume_aligned(x, ALIGN_SZ))
#define CFPTR(x) reinterpret_cast<float* __restrict__>(__builtin_assume_aligned(x, ALIGN_SZ))

alignas(ALIGN_SZ) static const float conj_values[] = {1.0, -1.0, 1.0, -1.0};

using sp_2_t = hn::FixedTag<MFFTELEMRI, 2>;
using sp_4_t = hn::FixedTag<MFFTELEMRI, 4>;
auto sp_2 = sp_2_t();
auto sp_4 = sp_4_t();

typedef struct {
  MFFTELEM* __restrict__ a_n;
  MFFTELEM* __restrict__ b_n;
  MFFTELEM* __restrict__ A_X;  // temp buffers
  MFFTELEM* __restrict__ B_X;  // temp buffers
  int64_t M;
  int64_t N;
  int32_t flags;
} bs_buffer_t;

// Global buffer for Bluestein algorithm
static bs_buffer_t bs_buff = {nullptr, nullptr, nullptr, nullptr, 0, 0, 0};

// Cleanup function to free global buffer
void free_bluestein_buffer(void) {
  if (bs_buff.N != 0) {
    free(bs_buff.a_n);
    free(bs_buff.b_n);
    free(bs_buff.A_X);
    free(bs_buff.B_X);
    bs_buff.N = 0;
    bs_buff.M = 0;
    bs_buff.a_n = nullptr;
    bs_buff.b_n = nullptr;
    bs_buff.A_X = nullptr;
    bs_buff.B_X = nullptr;
  }
}

static inline int64_t nextpow2_exp(uint64_t n) {
  if (n == 0) return 0;
  if (n == 1) return 2;
  int64_t high_bit = 63 - count_leading_zeros(n);
  if ((n & (n - 1)) == 0) return high_bit;
  return high_bit + 1;
}

/*
static const float LIMIT_FP = 1e20;

void trap_large_v(const MFFTELEM *__restrict__ x, int line)
{
  int64_t M = bs_buff.M;
  for (int i = 0; i < M; ++i)
  {
    MFFTELEM c = x[i];
    if ((std::abs(std::real(c)) > LIMIT_FP) || (std::abs(c) > LIMIT_FP))
    {
      std::cerr << "line: " << line << " large value in b_n at i=" << i << " c=" << c << std::endl;
      break;
    }
  }
}

void trap_large_value(std::complex<double> x, int line) {
   if (std::abs(std::real(x))>LIMIT_FP || std::abs(std::imag(x))>LIMIT_FP) {
      std::cerr << "large value: line=" << line << " M=" << bs_buff.M << " x=" << x << std::endl;
   }
}*/

template <bool Inverse>
void bluestein_init(int64_t N, int64_t M, int32_t flags) {
  bool init = (bs_buff.M == 0);
  if (!init) {
    if (bs_buff.M != M) {
      init = true;
    } else if ((bs_buff.flags & P_INVERSE) != Inverse) {
      // Conjugate b_n array
      MFFTELEM* b_n = bs_buff.b_n;
      for (int64_t i = 0; i < M; i++) {
        b_n[i] = conj(b_n[i]);
      }
      bs_buff.flags = flags;
    }
  }
  if (init) {
    // Free existing buffers if they exist
    if (bs_buff.b_n != nullptr) {
      free_bluestein_buffer();
    }
    // Allocate new buffers
    bs_buff.a_n = (MFFTELEM* __restrict__)minaligned_alloc(ALIGN_SZ, sizeof(MFFTELEM), M);
    bs_buff.b_n = (MFFTELEM* __restrict__)minaligned_calloc(ALIGN_SZ, sizeof(MFFTELEM), M);
    bs_buff.A_X = (MFFTELEM* __restrict__)minaligned_alloc(ALIGN_SZ, sizeof(MFFTELEM), M);
    bs_buff.B_X = (MFFTELEM* __restrict__)minaligned_alloc(ALIGN_SZ, sizeof(MFFTELEM), M);
    bs_buff.M = M;
    bs_buff.flags = flags;
  }

  if (init || bs_buff.N != N) {
    MFFTELEM* b_n = bs_buff.b_n;
    if (N < bs_buff.N) memset(b_n + N, 0, (M - 2 * N + 1) * sizeof(MFFTELEM));  // fill hole
    b_n[0] = (MFFTELEM)1.0;
    double arg = Inverse ? -M_PI / N : M_PI / N;
    std::complex<double> c_e;
    for (int64_t n = 1; n < N; n++) {
      c_e = minsincos(arg * n * n);
      b_n[n] = (MFFTELEM)c_e;
      b_n[M - n] = (MFFTELEM)c_e;
    }
    bs_buff.N = N;
  }
}

// Bluestein FFT implementation
template <bool Inverse>
void bluestein(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t discard_e1,
               const int64_t bp, const int64_t stride, const int32_t flags) {
  MFFTELEM* __restrict__ y = *YY;
  MFFTELEM* __restrict__ x = *XX;
  const int32_t e1 = nextpow2_exp(2 * N - 1);
  const int64_t M = 1 << e1;
  const auto conj_mask = hn::Load(sp_4, conj_values);
  if constexpr (Inverse)
    bluestein_init<true>(N, M, flags);
  else
    bluestein_init<false>(N, M, flags);
  MFFTELEM* a_n = bs_buff.a_n;
  const MFFTELEM* b_n = bs_buff.b_n;
  MFFTELEM* A_X = bs_buff.A_X;
  MFFTELEM* B_X = bs_buff.B_X;
  memset(a_n, 0, M * sizeof(MFFTELEM));
  for (int64_t n = 0; n < N - 1; n += 2) {
    auto bv = hn::Load(sp_4, CCFPTR(&b_n[n]));
    auto c = hn::Mul(bv, conj_mask);
    auto x_low = hn::Load(sp_2, CCFPTR(&x[bp + stride * n]));
    auto x_high = hn::Load(sp_2, CCFPTR(&x[bp + stride * (n + 1)]));
    auto xv = hn::Combine(sp_4, x_high, x_low);
    auto a = hn::MulComplex(xv, c);
    hn::Store(a, sp_4, CFPTR(&a_n[n]));
    hn::Store(hn::LowerHalf(c), sp_2, CFPTR(&y[bp + stride * n]));
    hn::Store(hn::UpperHalf(sp_4, c), sp_2, CFPTR(&y[bp + stride * (n + 1)]));
  }
  if (N & 1) {
    auto b = b_n[N - 1];
    auto c = std::conj(b);
    auto xv = x[bp + stride * (N - 1)];
    a_n[N - 1] = xv * c;
    y[bp + stride * (N - 1)] = c;
  }
  memcpy(B_X, b_n, M * sizeof(MFFTELEM));
  fftr2<false>(&A_X, &a_n, M, e1, 0, 1, P_NONE);
  fftr2<false>(&a_n, &B_X, M, e1, 0, 1, P_NONE);
  // M always power of 2
  for (int64_t i = 0; i < M - 1; i += 2) {
    auto AXv = hn::Load(sp_4, CCFPTR(&A_X[i]));
    auto anv = hn::Load(sp_4, CCFPTR(&a_n[i]));
    anv = hn::MulComplex(anv, AXv);
    hn::Store(anv, sp_4, CFPTR(&a_n[i]));
  }
  fftr2<true>(&B_X, &a_n, M, e1, 0, 1, P_INVERSE);
  bs_buff.A_X = A_X;  // rebind
  bs_buff.B_X = B_X;  // rebind
  bs_buff.a_n = a_n;  // rebind
  auto scale = hn::Set(sp_4, 1.0 / M);
  for (int64_t i = 0; i < N - 1; i += 2) {
    auto BXv = hn::Load(sp_4, CCFPTR(&B_X[i]));
    auto yv = hn::Load(sp_4, CCFPTR(&y[bp + stride * i]));
    yv = hn::MulComplex(yv, BXv);
    yv = hn::Mul(yv, scale);
    hn::Store(yv, sp_4, CFPTR(&y[bp + stride * i]));
  }
  if (N & 1) {
    auto BXv = B_X[N - 1];
    auto yv = y[bp + stride * (N - 1)];
    yv = (1.0f / M) * (yv * BXv);
    y[bp + stride * (N - 1)] = yv;
  }
}

template void bluestein_init<false>(int64_t N, int64_t M, int32_t flags);
template void bluestein_init<true>(int64_t N, int64_t M, int32_t flags);

template void bluestein<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                               const int32_t discard_e1, const int64_t bp, const int64_t stride,
                               const int32_t flags);
template void bluestein<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                              const int32_t discard_e1, const int64_t bp, const int64_t stride,
                              const int32_t flags);