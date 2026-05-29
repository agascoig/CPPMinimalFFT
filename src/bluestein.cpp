// Bluestein's FFT algorithm

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"

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
  if ((n & (n - 1)) == 0) {
    return high_bit;
  }

  return high_bit + 1;
}

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
        b_n[i] = conj(bs_buff.b_n[i]);
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
    bs_buff.a_n = (MFFTELEM* __restrict__)minaligned_alloc(sizeof(MFFTELEM), sizeof(MFFTELEM), M);
    bs_buff.b_n = (MFFTELEM* __restrict__)minaligned_calloc(sizeof(MFFTELEM), sizeof(MFFTELEM), M);
    bs_buff.A_X = (MFFTELEM* __restrict__)minaligned_alloc(sizeof(MFFTELEM), sizeof(MFFTELEM), M);
    bs_buff.B_X = (MFFTELEM* __restrict__)minaligned_alloc(sizeof(MFFTELEM), sizeof(MFFTELEM), M);
    bs_buff.M = M;
    bs_buff.flags = flags;
  }

  if (init || bs_buff.N != N) {
    MFFTELEM* b_n = bs_buff.b_n;
    if (N < bs_buff.N) {
      memset(b_n + N, 0, (M - 2 * N + 1) * sizeof(MFFTELEM));  // fill hole
    }
    b_n[0] = (MFFTELEM)1.0;
    double arg = Inverse ? -M_PI / N : M_PI / N;
    std::complex<double> c_e;
    for (int64_t n = 1; n < N; n++) {
      c_e = minsincos(arg * n * n);
      b_n[n] = (MFFTELEM)c_e;
      b_n[M - n] = (MFFTELEM)c_e;
    }
  }
  bs_buff.N = N;
}

// Bluestein FFT implementation
template <bool Inverse>
void bluestein(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t discard_e1,
               const int64_t bp, const int64_t stride, const int32_t flags) {
  MFFTELEM* __restrict__ y = *YY;
  MFFTELEM* __restrict__ x = *XX;

  const int32_t e1 = nextpow2_exp(2 * N - 1);
  const int64_t M = 1 << e1;

  if constexpr (Inverse)
    bluestein_init<true>(N, M, flags);
  else
    bluestein_init<false>(N, M, flags);

  MFFTELEM* a_n = bs_buff.a_n;
  const MFFTELEM* b_n = bs_buff.b_n;
  MFFTELEM* A_X = bs_buff.A_X;
  MFFTELEM* B_X = bs_buff.B_X;

  // Zero out a_n
  memset(a_n, 0, M * sizeof(MFFTELEM));

  // Fill a_n and y arrays
  for (int64_t n = 0; n < N; n++) {
    MFFTELEM c = std::conj(b_n[n]);
    a_n[n] = x[bp + stride * n] * c;
    y[bp + stride * n] = c;
  }

  // Copy b_n to B_X
  memcpy(B_X, b_n, M * sizeof(MFFTELEM));

  // Perform circular convolution using FFT
  fftr2<false>(&A_X, &a_n, M, e1, 0, 1, P_NONE);
  fftr2<false>(&a_n, &B_X, M, e1, 0, 1, P_NONE);  // Forward FFT of b_n
  for (int64_t i = 0; i < M; i++) {
    a_n[i] *= A_X[i];
  }
  fftr2<true>(&B_X, &a_n, M, e1, 0, 1, P_INVERSE);
  bs_buff.A_X = A_X;  // rebind
  bs_buff.B_X = B_X;  // rebind
  bs_buff.a_n = a_n;  // rebind
  // Scale by 1/M
  double scale = 1.0 / M;

  // Final multiplication
  for (int64_t i = 0; i < N; i++) {
    y[bp + stride * i] *= ((MFFTELEMRI)scale * B_X[i]);
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
