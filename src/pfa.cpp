
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
  int64_t buf[MAX_FACTORS * 2 - 1] = {0};
  int64_t* np = &buf[0];
  int64_t* R = &buf[MAX_FACTORS];  // only need nf-1

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

void nmap_2(MFFTELEM* __restrict__ Y, MFFTELEM* __restrict__ X, const int64_t bp,
            const int64_t stride, const int64_t N1, const int64_t N2, const int64_t Q1P) {
  int64_t Ns[] = {N1, N2};
  int64_t QP[] = {Q1P};
  nmap<2, MFFTELEM>(Y, X, bp, stride, Ns, QP);
}

void kmap_2(MFFTELEM* __restrict__ Y, MFFTELEM* __restrict__ X, const int64_t bp,
            const int64_t stride, const int64_t N1, const int64_t N2, const int64_t Q2P) {
  int64_t Ns[] = {N1, N2};
  int64_t Qs[] = {0, Q2P};
  kmap<2, MFFTELEM>(Y, X, bp, stride, Ns, Qs);
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
    if (i&1==0)
      do_fft<fft_func_t>(&X2D, &Y2D, Ns, es, bp, stride, flags, fs, nullptr, i);
    else
      do_fft<fft_func_t>(&Y2D, &X2D, Ns, es, bp, stride, flags, fs, nullptr, i);
  }

  if (nf&1==1) {
    Y = Y2D.data;
    X = X2D.data;
  }
  else {
    Y = X2D.data;
    X = Y2D.data;
  }

  kmap<nf, MFFTELEM>(Y, X, bp, stride, Ns, params);

  *YY = Y;
  *XX = X;
}

// Two-factor prime factor algorithm
void prime_factor_2(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;

  int64_t N1 = Ns[0];
  int64_t N2 = Ns[1];
  int64_t N = N1 * N2;

  nmap_2(Y, X, bp, stride, N1, N2, params[0]);

  // Create 2D arrays for FFT operations
  MDArray Y2D = create_mdarray(Y, Ns, 2);
  MDArray X2D = create_mdarray(X, Ns, 2);

  // Perform FFTs
  do_fft<fft_func_t>(&X2D, &Y2D, Ns, es, bp, stride, flags, fs, nullptr, 0);
  do_fft<fft_func_t>(&Y2D, &X2D, Ns, es, bp, stride, flags, fs, nullptr, 1);

  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  kmap_2(X, Y, bp, stride, N1, N2, params[1]);

  *YY = X;
  *XX = Y;
}

void nmap_3(MFFTELEM* __restrict__ Y, MFFTELEM* __restrict__ X, const int64_t bp,
            const int64_t stride, const int64_t N1, const int64_t N2, const int64_t N3,
            const int64_t Q1P, const int64_t Q2P) {
  int64_t Ns[3] = {N1, N2, N3};
  int64_t Qs[2] = {Q1P, Q2P};
  nmap<3, MFFTELEM>(Y, X, bp, stride, Ns, Qs);
}

void kmap_3(MFFTELEM* __restrict__ Y, MFFTELEM* __restrict__ X, const int64_t bp,
            const int64_t stride, const int64_t N1, const int64_t N2, const int64_t N3,
            const int64_t Q3P, const int64_t Q4P) {
  int64_t Ns[3] = {N1, N2, N3};
  int64_t Qs[4] = {0, 0, Q3P, Q4P};
  kmap<3, MFFTELEM>(Y, X, bp, stride, Ns, Qs);
}

// Three-factor prime factor algorithm
void prime_factor_3(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                    const int64_t bp, const int64_t stride, const int32_t flags,
                    const fft_func_t* fs, const int64_t* params) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;

  const int64_t N1 = Ns[0];
  const int64_t N2 = Ns[1];
  const int64_t N3 = Ns[2];
  const int64_t N = N1 * N2 * N3;

  // Forward mapping
  nmap_3(Y, X, bp, stride, N1, N2, N3, params[0], params[1]);

  // Create 3D arrays for FFT operations
  MDArray Y123 = create_mdarray(Y, Ns, 3);
  MDArray X123 = create_mdarray(X, Ns, 3);

  // Perform FFTs
  do_fft<fft_func_t>(&X123, &Y123, Ns, es, bp, stride, flags, fs, nullptr, 0);
  do_fft<fft_func_t>(&Y123, &X123, Ns, es, bp, stride, flags, fs, nullptr, 1);
  do_fft<fft_func_t>(&X123, &Y123, Ns, es, bp, stride, flags, fs, nullptr, 2);
  // Rebind
  Y = Y123.data;
  X = X123.data;

  kmap_3(Y, X, bp, stride, N1, N2, N3, params[2], params[3]);

  *YY = Y;
  *XX = X;
}

void generate_pfa_params(int32_t factor_count, const int64_t* Ns, int64_t* params) {
  switch (factor_count) {
    case 2:
    case 3:
      Qs(factor_count, Ns, params);
      break;
    case 4: {
      const int64_t NsE[] = {Ns[0] * Ns[1], Ns[2] * Ns[3]};
      Qs(2, NsE, params);
      Qs(2, Ns, params + 2);
      Qs(2, Ns + 2, params + 4);
    } break;
    case 5: {
      const int64_t NsE[] = {Ns[0] * Ns[1] * Ns[2], Ns[3] * Ns[4]};
      Qs(2, NsE, params);
      Qs(3, Ns, params + 2);
      Qs(2, Ns + 3, params + 6);
    } break;
    case 6: {
      const int64_t NsE[] = {Ns[0] * Ns[1] * Ns[2], Ns[3] * Ns[4] * Ns[5]};
      Qs(2, NsE, params);
      Qs(3, Ns, params + 2);
      Qs(3, Ns + 3, params + 6);
    } break;
    default:
      minassert(0, "generate_pfa_params only supports 2-6 factors");
  }
}

void pfa_extend_4(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                  const int64_t bp, const int64_t stride, const int32_t flags, const fft_func_t* fs,
                  const int64_t* params) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;

  const int64_t NsE[] = {Ns[0] * Ns[1], Ns[2] * Ns[3]};

  nmap_2(Y, X, bp, stride, NsE[0], NsE[1], params[0]);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa2_t>(&X2D, &Y2D, Ns, es, bp, stride, flags, fs, params + 2, 0);
  do_fft<pfa2_t>(&Y2D, &X2D, Ns + 2, es + 2, bp, stride, flags, fs + 2, params + 4, 1);

  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  kmap_2(X, Y, bp, stride, NsE[0], NsE[1], params[1]);

  *YY = X;
  *XX = Y;
}

void pfa_extend_5(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                  const int64_t bp, const int64_t stride, const int32_t flags, const fft_func_t* fs,
                  const int64_t* params) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;

  const int64_t NsE[] = {Ns[0] * Ns[1] * Ns[2], Ns[3] * Ns[4]};

  nmap_2(Y, X, bp, stride, NsE[0], NsE[1], params[0]);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa3_t>(&X2D, &Y2D, Ns, es, bp, stride, flags, fs, params + 2, 0);
  do_fft<pfa2_t>(&Y2D, &X2D, Ns + 3, es + 3, bp, stride, flags, fs + 3, params + 6, 1);

  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  kmap_2(X, Y, bp, stride, NsE[0], NsE[1], params[1]);

  *YY = X;
  *XX = Y;
}

void pfa_extend_6(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                  const int64_t bp, const int64_t stride, const int32_t flags, const fft_func_t* fs,
                  const int64_t* params) {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;

  const int64_t NsE[] = {Ns[0] * Ns[1] * Ns[2], Ns[3] * Ns[4] * Ns[5]};

  nmap_2(Y, X, bp, stride, NsE[0], NsE[1], params[0]);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa3_t>(&X2D, &Y2D, Ns, es, bp, stride, flags, fs, params + 2, 0);
  do_fft<pfa3_t>(&Y2D, &X2D, Ns + 3, es + 3, bp, stride, flags, fs + 3, params + 6, 1);

  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  kmap_2(X, Y, bp, stride, NsE[0], NsE[1], params[1]);

  *YY = X;
  *XX = Y;
}