
#include <cassert>
#include <cstdlib>
#include <cstring>

#include "CPPMinimalFFT.hpp"

// Extended Euclidean algorithm
typedef struct {
  int64_t g, x, y;
} ExtendedEuclidResult;

ExtendedEuclidResult extended_euclid(int64_t a, int64_t b) {
  minassert(a >= 0 && b >= 0, "a and b must be non-negative");

  if (a == 0) {
    ExtendedEuclidResult result = {b, 0, 1};
    return result;
  }

  ExtendedEuclidResult sub_result = extended_euclid(b % a, a);
  ExtendedEuclidResult result = {sub_result.g, sub_result.y - (b / a) * sub_result.x, sub_result.x};
  return result;
}

void Qs(int nf, const int64_t* Ns, int64_t* params) {
  int64_t N = 1;
  int64_t y;
  ExtendedEuclidResult r;
  for (int i = nf - 1; i > 0; --i) {
    N = N * Ns[i];
    r = extended_euclid(Ns[i - 1], N);
    y = r.y % Ns[i - 1];
    if (y < 0) y += Ns[i - 1];
    params[i - 1] = y;
  }
  N = 1;
  for (int i = 2; i <= nf; ++i) {
    N *= Ns[i - 2];
    r = extended_euclid(Ns[i - 1], N);
    y = r.y % Ns[i - 1];
    if (y < 0) y += Ns[i - 1];
    params[2 * nf - i - 1] = y;
  }
}

// Inline branchless mask mux mod function
static inline int64_t mask_mux_mod(int64_t a, int64_t B) { return a - (B & -(a >= B)); }

template <int nf, typename T>
void nmap(T* __restrict__ Y, T* __restrict__ X, const int64_t bp, const int64_t stride,
          const int64_t* Ns, const int64_t* QP) {
  int64_t buf[nf * 2 - 1] = {0};
  int64_t* np = &buf[0];
  int64_t* R = &buf[nf];  // only need nf-1

  int64_t n, N;
  int64_t rhs_n_stride = bp;

  int64_t np_p = nf - 1;

  while (1) {
    n = 0;
    N = 1;
    for (int i = 0; i < (nf - 1); ++i) {
      n += N * mask_mux_mod(np[i] + R[i], Ns[i]);
      R[i] = mask_mux_mod(R[i] + QP[i], Ns[i]);
      N = N * Ns[i];
    }
    n += N * np[nf - 1];
    Y[bp + stride * n] = X[rhs_n_stride];
    rhs_n_stride += stride;

    np_p = nf - 1;
    while (np_p >= 0) {
      np[np_p]++;
      if (np[np_p] == Ns[np_p]) {
        if (np_p == 0) return;
        np[np_p] = 0;
        R[np_p - 1] = 0;
      } else {
        break;
      }
      np_p--;
    }
  }
}

template <int nf, typename T>
void kmap(T* __restrict__ Y, T* __restrict__ X, const int64_t bp, const int64_t stride,
          const int64_t* Ns, const int64_t* QP) {
  int64_t buf[MAX_FACTORS * 2 - 1] = {0};
  int64_t* np = &buf[0];
  int64_t* R = &buf[MAX_FACTORS];

  int64_t k, N;
  int64_t lhs_k_stride = bp;

  int64_t np_p = nf - 1;

  while (1) {
    k = np[0];
    N = Ns[0];
    for (int i = 1; i < nf; ++i) {
      k += N * mask_mux_mod(np[i] + R[i - 1], Ns[i]);
      R[i - 1] = mask_mux_mod(R[i - 1] + QP[2 * nf - 2 - i], Ns[i]);
      N = N * Ns[i];
    }
    Y[lhs_k_stride] = X[bp + stride * k];
    lhs_k_stride += stride;

    np_p = 0;
    while (np_p < nf) {
      np[np_p]++;
      if (np[np_p] == Ns[np_p]) {
        if (np_p == (nf - 1)) return;
        np[np_p] = 0;
        R[np_p] = 0;
      } else {
        break;
      }
      np_p++;
    }
  }
}

template <int nf>
void prime_factor(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  
  nmap<nf, MFFTELEM>(Y, X, bp, stride, Ns, params);

  MDArray YMD = create_mdarray(Y, Ns, nf);
  MDArray XMD = create_mdarray(X, Ns, nf);

  for (int i=0;i<nf;++i) {
    if ((i&1)==0)
      do_fft(&XMD, &YMD, Ns, es, bp, stride, flags, fs, nullptr, i);
    else
      do_fft(&YMD, &XMD, Ns, es, bp, stride, flags, fs, nullptr, i);
  }

  if ((nf&1)==1) {
    Y = YMD.data;
    X = XMD.data;
  }
  else {
    Y = XMD.data;
    X = YMD.data;
  }

  kmap<nf, MFFTELEM>(Y, X, bp, stride, Ns, params);

  *YY = Y;
  *XX = X;
}

void generate_pfa_params(int32_t factor_count, const int64_t* Ns, int64_t* params) {
  Qs(factor_count, Ns, params);
}

template void prime_factor<2>(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params);

template void prime_factor<3>(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params);

template void prime_factor<4>(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params);

template void prime_factor<5>(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params);

template void prime_factor<6>(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params);

template void prime_factor<7>(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params);


