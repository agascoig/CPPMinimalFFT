// For approx N<20, the direct DFT can be as fast as the FFT
// due to lower communication cost. Also useful for testing.

#include <hwy/highway.h>

#include <complex>
#include <cstdlib>

#include "../CPPMinimalFFT.hpp"
#include "../plan.hpp"
#include "../weights.hpp"

namespace hn = hwy::HWY_NAMESPACE;

#define CCDPTR(x) reinterpret_cast<const double* __restrict__ __attribute__((aligned(ALIGN_SZ)))>(x)
#define CDPTR(x) reinterpret_cast<double* __restrict__ __attribute__((aligned(ALIGN_SZ)))>(x)
#define CCFPTR(x) reinterpret_cast<const float* __restrict__>(__builtin_assume_aligned(x, ALIGN_SZ))
#define CFPTR(x) reinterpret_cast<float* __restrict__>(__builtin_assume_aligned(x, ALIGN_SZ))

alignas(ALIGN_SZ) static const double conj_values[] = {1.0, -1.0};

using sp_2_t = hn::FixedTag<float, 2>;
using sp_4_t = hn::FixedTag<float, 4>;

static auto sp_2 = sp_2_t();
static auto sp_4 = sp_4_t();

static inline int32_t mask_mux_mod(const int32_t a, const int32_t B) { return a - (B & -(a >= B)); }

static inline int32_t rev_mask_mux_mod(const int32_t a, const int32_t B) {
  return a + (B & -(a < 0));
}

// Direct DFT implementation: good for small N<=DIRECT_SZ
template <bool Inverse>
void direct_dft(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1, const int64_t bp,
                const int64_t stride, const int32_t flags) {
  MFFTELEM* __restrict__ y = *YY;
  MFFTELEM* __restrict__ x = *XX;
  minassert(N > 0 && N <= DIRECT_SZ, "N too large for direct DFT");
  const auto* __restrict__ W = reinterpret_cast<const std::complex<MFFTELEMRI>*>(DIRECT_COEFFS[N]);
  auto s = hn::Set(sp_4, 0.0);

  for (int32_t k = 0; k < N; k++) {
    int32_t nk_mod0, nk_mod1;
    if constexpr (Inverse) {
      nk_mod0 = 0;
      nk_mod1 = rev_mask_mux_mod(nk_mod0 - k, N);
    } else {
      nk_mod0 = 0;
      nk_mod1 = mask_mux_mod(nk_mod0 + k, N);
    }
    int64_t step = bp;
    s = hn::Set(sp_4, 0.0);
    for (int32_t n = 0; n < N - 1; n += 2) {
      auto Wv_low = hn::Load(sp_2, CCFPTR(&W[nk_mod0]));
      auto Wv_high = hn::Load(sp_2, CCFPTR(&W[nk_mod1]));
      auto Wv = hn::Combine(sp_4, Wv_high, Wv_low);
      auto x_low = hn::Load(sp_2, CCFPTR(&x[step]));
      auto x_high = hn::Load(sp_2, CCFPTR(&x[step + stride]));
      auto xv = hn::Combine(sp_4, x_high, x_low);
      s = hn::Add(s, hn::MulComplex(Wv, xv));
      if constexpr (Inverse) {
        nk_mod1 = rev_mask_mux_mod(nk_mod1 - k, N);
        nk_mod0 = nk_mod1;
        nk_mod1 = rev_mask_mux_mod(nk_mod1 - k, N);
      } else {
        nk_mod1 = mask_mux_mod(nk_mod1 + k, N);
        nk_mod0 = nk_mod1;
        nk_mod1 = mask_mux_mod(nk_mod1 + k, N);
      }
      step += 2 * stride;
    }
    auto sc = hn::Set(sp_2, 0.0);
    if (N & 1) {
      auto v = hn::Load(sp_2, CCFPTR(&W[nk_mod0]));
      sc = hn::Load(sp_2, CCFPTR(&x[step]));
      sc = hn::MulComplex(v, sc);
    }
    sc = hn::Add(sc, hn::LowerHalf(s));
    sc = hn::Add(sc, hn::UpperHalf(sp_4, s));
    hn::Store(sc, sp_2, CFPTR(&y[bp + stride * k]));
  }
}

template void direct_dft<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
                                const int64_t bp, const int64_t stride, const int32_t flags);
template void direct_dft<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
                               const int64_t bp, const int64_t stride, const int32_t flags);
