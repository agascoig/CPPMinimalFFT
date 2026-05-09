// For approx N<20, the direct DFT can be as fast as the FFT
// due to lower communication cost. Also useful for testing.

#include <complex>
#include <cstdlib>

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"
#include "weights.hpp"

static inline int32_t mask_mux_mod(const int32_t a, const int32_t B) { return a - (B & -(a >= B)); }

static inline int32_t rev_mask_mux_mod(const int32_t a, const int32_t B) {
  return a + (B & -(a < 0));
}

// Direct DFT implementation: good for small N<=DIRECT_SZ
template <bool Inverse>
void direct_dft(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
                const int64_t stride, const int32_t flags) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;

  minassert(N > 0 && N <= DIRECT_SZ, "N too large for direct DFT");

  auto* __restrict__ W =
      reinterpret_cast<const std::complex<MFFTELEMRI>* __restrict__>(DIRECT_COEFFS[N]);

  MFFTELEM s{};  // zero

  // special case Y[0]
  int64_t step = bp;
  for (int64_t n = 0; n < N; n++) {
    s += X[step];
    step += stride;
  }
  Y[bp] = s;

  if constexpr (Inverse) {
    for (int32_t k = 1; k < N; k++) {
      s = X[bp];
      step = bp + stride;

      int32_t nk_mod = N - k;
      for (int32_t n = 1; n < N; n++) {
        s = s + ((MFFTELEM)W[nk_mod] * X[step]);
        nk_mod = rev_mask_mux_mod(nk_mod - k, N);
        step += stride;
      }
      Y[bp + stride * k] = s;
    }
  }else {
    for (int32_t k = 1; k < N; k++) {
      s = X[bp];
      step = bp + stride;

      int32_t nk_mod = k;
      for (int32_t n = 1; n < N; n++) {
        s = s + ((MFFTELEM)W[nk_mod] * X[step]);
        nk_mod = mask_mux_mod(nk_mod + k, N);
        step += stride;
      }
      Y[bp + stride * k] = s;
    }
  }
}

template void direct_dft<false>(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
                 const int64_t bp, const int64_t stride, const int32_t flags);
template void direct_dft<true>(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
                 const int64_t bp, const int64_t stride, const int32_t flags);
