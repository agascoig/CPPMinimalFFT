// For approx N<20, the direct DFT can be as fast as the FFT
// due to lower communication cost. Also useful for testing.

#include <hwy/highway.h>
#include <cstdlib>

#include <complex>

#include "../CPPMinimalFFT.hpp"
#include "../plan.hpp"
#include "../weights.hpp"

namespace hn = hwy::HWY_NAMESPACE;

#define CCDPTR(x) \
  reinterpret_cast<const double *__restrict__ __attribute__((aligned(16)))>(x)
#define CDPTR(x) \
  reinterpret_cast<double *__restrict__ __attribute__((aligned(16)))>(x)

alignas(sizeof(double) * 2) static const double conj_values[] = {1.0, -1.0};

using D = hn::CappedTag<double, 2>;

static inline int32_t mask_mux_mod(const int32_t a, const int32_t B) {
  return a - (B & -(a >= B));
}

static inline int32_t rev_mask_mux_mod(const int32_t a, const int32_t B) {
  return a + (B & -(a < 0));
}

// Direct DFT implementation: good for small N<=DIRECT_SZ
void direct_dft(MFFTELEM **YY, MFFTELEM **XX, const int64_t N, const int32_t e1,
                const int64_t bp, const int64_t stride, const int32_t flags) {
  MFFTELEM *__restrict__ y = *YY;
  MFFTELEM *__restrict__ x = *XX;
  minassert(N > 0 && N <= DIRECT_SZ, "N too large for direct DFT");
  D d;
  const auto *__restrict__ W =
      reinterpret_cast<const std::complex<double> *>(DIRECT_COEFFS[N]);
  const bool inverse = (flags & P_INVERSE);
  auto s = hn::Set(d, 0.0);
  int64_t step = bp;
  for (int64_t n = 0; n < N; n++) {
    auto v = hn::Load(d, CCDPTR(&x[step]));
    s = hn::Add(s, v);
    step += stride;
  }
  hn::Store(s, d, CDPTR(&y[bp]));
  if (inverse) {
    for (int32_t k = 1; k < N; k++) {
      s = hn::Load(d, CDPTR(&x[bp]));
      step = bp + stride;
      int32_t nk_mod = N - k;
      for (int32_t n = 1; n < N; n++) {
        auto Wv = hn::Load(d, CCDPTR(&W[nk_mod]));
        auto xv = hn::Load(d, CCDPTR(&x[step]));
        s = hn::Add(s, hn::MulComplex(Wv, xv));
        nk_mod = rev_mask_mux_mod(nk_mod - k, N);
        step += stride;
      }
      hn::Store(s, d, CDPTR(&y[bp + stride * k]));
    }
  } else {
    for (int32_t k = 1; k < N; k++) {
      s = hn::Load(d, CDPTR(&x[bp]));
      step = bp + stride;
      int32_t nk_mod = k;
      int n = 1;
      for (n = 1; n < N; n++) {
        auto Wv = hn::Load(d, CCDPTR(&W[nk_mod]));
        auto xv = hn::Load(d, CCDPTR(&x[step]));
        s = hn::Add(s, hn::MulComplex(Wv, xv));
        nk_mod = mask_mux_mod(nk_mod + k, N);
        step += stride;
      }
      hn::Store(s, d, CDPTR(&y[bp + stride * k]));
    }
  }
}