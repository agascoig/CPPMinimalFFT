
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
  ExtendedEuclidResult result = {
      sub_result.g, sub_result.y - (b / a) * sub_result.x, sub_result.x};
  return result;
}

void pfa_params_2(const int64_t *Ns, int64_t *params) {
  const int64_t N1 = Ns[0];
  const int64_t N2 = Ns[1];

  ExtendedEuclidResult result = extended_euclid(N1, N2);
  minassert(result.g == 1, "prime_factor N1 and N2 must be coprime");

  int64_t M1 = result.x;
  int64_t M2 = result.y;

  int64_t Q1P = M2 % N1;
  if (Q1P < 0) Q1P += N1;

  int64_t Q2P = M1 % N2;
  if (Q2P < 0) Q2P += N2;

  params[0] = Q1P;
  params[1] = Q2P;
}

void pfa_params_3(const int64_t *Ns, int64_t *params) {
  const int64_t N1 = Ns[0];
  const int64_t N2 = Ns[1];
  const int64_t N3 = Ns[2];

  ExtendedEuclidResult r1 = extended_euclid(N1, N2 * N3);
  ExtendedEuclidResult r2 = extended_euclid(N2, N1 * N3);
  ExtendedEuclidResult r3 = extended_euclid(N3, N1 * N2);
  ExtendedEuclidResult r4 = extended_euclid(N2 * N3, N1);

  minassert(r1.g == 1 && r2.g == 1 && r3.g == 1 && r4.g == 1,
            "prime_factor N1, N2, N3 must be coprime");

  int64_t Q1 = r1.y;
  int64_t Q2 = r2.y * N1;
  int64_t Q3 = r3.y;
  int64_t Q4 = r4.y;

  int64_t Q1P = Q1 % N1;
  if (Q1P < 0) Q1P += N1;

  int64_t Q2P = Q2 % N2;
  if (Q2P < 0) Q2P += N2;

  int64_t P1 = Q4 % N2;
  if (P1 < 0) P1 += N2;

  int64_t P2 = Q3 % N3;
  if (P2 < 0) P2 += N3;

  params[0] = Q1P;
  params[1] = Q2P;
  params[2] = P1;
  params[3] = P2;
}

void generate_pfa_params(int32_t factor_count, const int64_t *Ns,
                         int64_t *params) {
  switch (factor_count) {
    case 2:
      pfa_params_2(Ns, params);
      break;
    case 3:
      pfa_params_3(Ns, params);
      break;
    case 4: {
      const int64_t NsE[] = {Ns[0] * Ns[1], Ns[2] * Ns[3]};
      pfa_params_2(NsE, params);
      pfa_params_2(Ns, params + 2);
      pfa_params_2(Ns + 2, params + 4);
    } break;
    case 5: {
      const int64_t NsE[] = {Ns[0] * Ns[1] * Ns[2], Ns[3] * Ns[4]};
      pfa_params_2(NsE, params);
      pfa_params_3(Ns, params + 2);
      pfa_params_2(Ns + 3, params + 6);
    } break;
    case 6: {
      const int64_t NsE[] = {Ns[0] * Ns[1] * Ns[2], Ns[3] * Ns[4] * Ns[5]};
      pfa_params_2(NsE, params);
      pfa_params_3(Ns, params + 2);
      pfa_params_3(Ns + 3, params + 6);
    } break;
    default:
      minassert(0, "generate_pfa_params only supports 2-6 factors");
  }
}

// Inline mask mux mod function
static inline int64_t mask_mux_mod(int64_t a, int64_t B) {
  return a - (B & -(a >= B));
}

static inline void nmap_2(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                          const int64_t bp, const int64_t stride,
                          const int64_t N1, const int64_t N2,
                          const int64_t Q1P) {
  // Forward mapping
  int64_t rhs_n_stride = bp;
  int64_t L2 = 0;
  for (int64_t n1p = 0; n1p < N1; n1p++) {
    int64_t R1 = 0;
    L2 = 0;
    for (int64_t n2p = 0; n2p < N2; n2p++) {
      int64_t n1 = mask_mux_mod(n1p + R1, N1);
      int64_t lhs_n = n1 + L2;
      Y[bp + stride * lhs_n] = X[rhs_n_stride];
      R1 = mask_mux_mod(R1 + Q1P, N1);
      rhs_n_stride += stride;
      L2 += N1;
    }
  }
}

static inline void kmap_2(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                          const int64_t bp, const int64_t stride,
                          const int64_t N1, const int64_t N2,
                          const int64_t Q2P) {
  int64_t lhs_k_stride = bp;
  for (int64_t k2p = 0; k2p < N2; k2p++) {
    int64_t R1 = 0;
    for (int64_t k1p = 0; k1p < N1; k1p++) {
      int64_t k2 = mask_mux_mod(k2p + R1, N2);
      int64_t rhs_k = k1p + k2 * N1;
      Y[lhs_k_stride] = X[bp + stride * rhs_k];
      R1 = mask_mux_mod(R1 + Q2P, N2);
      lhs_k_stride += stride;
    }
  }
}
// Two-factor prime factor algorithm
void prime_factor_2(MFFTELEM **YY, MFFTELEM **XX, const int64_t *Ns,
                    const int32_t *es, const int64_t bp, const int64_t stride,
                    const int32_t flags, const fft_func_t *fs,
                    const int64_t *params) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

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

// Forward n-mapping for 3-factor
static inline void nmap_3(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                          const int64_t bp, const int64_t stride,
                          const int64_t N1, const int64_t N2, const int64_t N3,
                          const int64_t Q1P, const int64_t Q2P) {
  int64_t rhs_n_stride = bp;
  for (int64_t n1p = 0; n1p < N1; n1p++) {
    int64_t R1 = 0;
    for (int64_t n2p = 0; n2p < N2; n2p++) {
      int64_t R2 = 0;
      for (int64_t n3p = 0; n3p < N3; n3p++) {
        int64_t n1 = mask_mux_mod(n1p + R1, N1);
        int64_t n2 = mask_mux_mod(n2p + R2, N2);
        int64_t lhs_n = n1 + N1 * n2 + N1 * N2 * n3p;
        Y[bp + stride * lhs_n] = X[rhs_n_stride];
        R1 = mask_mux_mod(R1 + Q1P, N1);
        R2 = mask_mux_mod(R2 + Q2P, N2);
        rhs_n_stride += stride;
      }
    }
  }
}

// Backward k-mapping for 3-factor
static inline void kmap_3(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X,
                          const int64_t bp, const int64_t stride,
                          const int64_t N1, const int64_t N2, const int64_t N3,
                          const int64_t P1, const int64_t P2) {
  int64_t lhs_k_stride = bp;
  for (int64_t k3p = 0; k3p < N3; k3p++) {
    int64_t R2 = 0;
    for (int64_t k2p = 0; k2p < N2; k2p++) {
      int64_t R1 = 0;
      for (int64_t k1p = 0; k1p < N1; k1p++) {
        int64_t k2 = mask_mux_mod(k2p + R1, N2);
        int64_t k3 = mask_mux_mod(k3p + R2, N3);
        int64_t rhs_k = k1p + N1 * k2 + N1 * N2 * k3;
        Y[lhs_k_stride] = X[bp + stride * rhs_k];
        R1 = mask_mux_mod(R1 + P1, N2);
        R2 = mask_mux_mod(R2 + P2, N3);
        lhs_k_stride += stride;
      }
    }
  }
}

// Three-factor prime factor algorithm
void prime_factor_3(MFFTELEM **YY, MFFTELEM **XX, const int64_t *Ns,
                    const int32_t *es, const int64_t bp, const int64_t stride,
                    const int32_t flags, const fft_func_t *fs,
                    const int64_t *params) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

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

void pfa_extend_4(MFFTELEM **YY, MFFTELEM **XX, const int64_t *Ns,
                  const int32_t *es, const int64_t bp, const int64_t stride,
                  const int32_t flags, const fft_func_t *fs,
                  const int64_t *params) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int64_t NsE[] = {Ns[0] * Ns[1], Ns[2] * Ns[3]};

  nmap_2(Y, X, bp, stride, NsE[0], NsE[1], params[0]);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa2_t>(&X2D, &Y2D, Ns, es, bp, stride, flags, fs, params + 2, 0);
  do_fft<pfa2_t>(&Y2D, &X2D, Ns + 2, es + 2, bp, stride, flags, fs + 2,
                 params + 4, 1);

  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  kmap_2(X, Y, bp, stride, NsE[0], NsE[1], params[1]);

  *YY = X;
  *XX = Y;
}

void pfa_extend_5(MFFTELEM **YY, MFFTELEM **XX, const int64_t *Ns,
                  const int32_t *es, const int64_t bp, const int64_t stride,
                  const int32_t flags, const fft_func_t *fs,
                  const int64_t *params) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int64_t NsE[] = {Ns[0] * Ns[1] * Ns[2], Ns[3] * Ns[4]};

  nmap_2(Y, X, bp, stride, NsE[0], NsE[1], params[0]);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa3_t>(&X2D, &Y2D, Ns, es, bp, stride, flags, fs, params + 2, 0);
  do_fft<pfa2_t>(&Y2D, &X2D, Ns + 3, es + 3, bp, stride, flags, fs + 3,
                 params + 6, 1);

  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  kmap_2(X, Y, bp, stride, NsE[0], NsE[1], params[1]);

  *YY = X;
  *XX = Y;
}

void pfa_extend_6(MFFTELEM **YY, MFFTELEM **XX, const int64_t *Ns,
                  const int32_t *es, const int64_t bp, const int64_t stride,
                  const int32_t flags, const fft_func_t *fs,
                  const int64_t *params) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int64_t NsE[] = {Ns[0] * Ns[1] * Ns[2], Ns[3] * Ns[4] * Ns[5]};

  nmap_2(Y, X, bp, stride, NsE[0], NsE[1], params[0]);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa3_t>(&X2D, &Y2D, Ns, es, bp, stride, flags, fs, params + 2, 0);
  do_fft<pfa3_t>(&Y2D, &X2D, Ns + 3, es + 3, bp, stride, flags, fs + 3,
                 params + 6, 1);

  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  kmap_2(X, Y, bp, stride, NsE[0], NsE[1], params[1]);

  *YY = X;
  *XX = Y;
}