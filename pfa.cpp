
#include "CPPMinimalFFT.hpp"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

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

// Inline mask mux mod function
static inline int64_t mask_mux_mod(int64_t a, int64_t B) {
  return a - (B & -(a >= B));
}

void nmap_2(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X, int64_t bp,
            int64_t stride, int64_t N1, int64_t N2, int64_t Q1P) {
  // Forward mapping
  int64_t rhs_n = 0;
  int64_t L2 = 0;
  for (int64_t n1p = 0; n1p < N1; n1p++) {
    int64_t R1 = 0;
    L2 = 0;
    for (int64_t n2p = 0; n2p < N2; n2p++) {
      int64_t n1 = mask_mux_mod(n1p + R1, N1);
      int64_t lhs_n = n1 + L2;
      Y[bp + stride * lhs_n] = X[bp + stride * rhs_n];
      R1 = mask_mux_mod(R1 + Q1P, N1);
      rhs_n++;
      L2 += N1;
    }
  }
}

void kmap_2(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X, int64_t bp,
            int64_t stride, int64_t N1, int64_t N2, int64_t Q2P) {
  int64_t lhs_k = 0;
  for (int64_t k2p = 0; k2p < N2; k2p++) {
    int64_t R1 = 0;
    for (int64_t k1p = 0; k1p < N1; k1p++) {
      int64_t k2 = mask_mux_mod(k2p + R1, N2);
      int64_t rhs_k = k1p + k2 * N1;
      Y[bp + stride * lhs_k] = X[bp + stride * rhs_k];
      R1 = mask_mux_mod(R1 + Q2P, N2);
      lhs_k++;
    }
  }
}
// Two-factor prime factor algorithm
void prime_factor_2(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                    const fft_func_t *fs, int64_t bp, int64_t stride,
                    int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  int64_t N1 = Ns[0];
  int64_t N2 = Ns[1];
  int64_t N = N1 * N2;

  ExtendedEuclidResult result = extended_euclid(N1, N2);
  minassert(result.g == 1, "prime_factor N1 and N2 must be coprime");

  int64_t M1 = result.x;
  int64_t M2 = result.y;

  int64_t Q1P = M2 % N1;
  if (Q1P < 0)
    Q1P += N1;

  nmap_2(Y, X, bp, stride, N1, N2, Q1P);

  // Create 2D arrays for FFT operations
  MDArray Y2D = create_mdarray(Y, Ns, 2);
  MDArray X2D = create_mdarray(X, Ns, 2);

  // Perform FFTs
  do_fft<fft_func_t>(&X2D, &Y2D, fs, Ns, 2, es, 0, bp, stride,
                     flags);
  do_fft<fft_func_t>(&Y2D, &X2D, fs, Ns, 2, es, 1, bp, stride,
                     flags);

  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  // Backward mapping
  int64_t Q2P = M1 % N2;
  if (Q2P < 0)
    Q2P += N2;

  kmap_2(X, Y, bp, stride, N1, N2, Q2P);

  *YY = X;
  *XX = Y;
}

// Structure for three-factor Q values
typedef struct {
  int64_t p1, p2, p3, p4;
  int64_t Q1, Q2, Q3, Q4;
} QValues;

QValues compute_Qs(int64_t N1, int64_t N2, int64_t N3) {
  ExtendedEuclidResult r1 = extended_euclid(N1, N2 * N3);
  ExtendedEuclidResult r2 = extended_euclid(N2, N1 * N3);
  ExtendedEuclidResult r3 = extended_euclid(N3, N1 * N2);
  ExtendedEuclidResult r4 = extended_euclid(N2 * N3, N1);

  minassert(r1.g == 1 && r2.g == 1 && r3.g == 1 && r4.g == 1,
            "N1, N2, N3 must be coprime");

  QValues q = {r1.x, r2.x, r3.x, r4.x, -r1.y, -r2.y * N1, -r3.y * N1, -r4.y};
  return q;
}

// Forward n-mapping for 3-factor
void nmap_3(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X, int64_t bp,
            int64_t stride, int64_t N1, int64_t N2, int64_t N3, int64_t Q1P,
            int64_t Q2P) {
  int64_t rhs_n = 0;
  for (int64_t n1p = 0; n1p < N1; n1p++) {
    int64_t R1 = 0;
    for (int64_t n2p = 0; n2p < N2; n2p++) {
      int64_t R2 = 0;
      for (int64_t n3p = 0; n3p < N3; n3p++) {
        int64_t n1 = mask_mux_mod(n1p + R1, N1);
        int64_t n2 = mask_mux_mod(n2p + R2, N2);
        int64_t lhs_n = n1 + N1 * n2 + N1 * N2 * n3p;
        Y[bp + stride * lhs_n] = X[bp + stride * rhs_n];
        R1 = mask_mux_mod(R1 + Q1P, N1);
        R2 = mask_mux_mod(R2 + Q2P, N2);
        rhs_n++;
      }
    }
  }
}

// Backward k-mapping for 3-factor
void kmap_3(MFFTELEM *__restrict__ Y, MFFTELEM *__restrict__ X, int64_t bp,
            int64_t stride, int64_t N1, int64_t N2, int64_t N3, int64_t P1,
            int64_t P2) {
  int64_t lhs_k = 0;
  for (int64_t k3p = 0; k3p < N3; k3p++) {
    int64_t R2 = 0;
    for (int64_t k2p = 0; k2p < N2; k2p++) {
      int64_t R1 = 0;
      for (int64_t k1p = 0; k1p < N1; k1p++) {
        int64_t k2 = mask_mux_mod(k2p + R1, N2);
        int64_t k3 = mask_mux_mod(k3p + R2, N3);
        int64_t rhs_k = k1p + N1 * k2 + N1 * N2 * k3;
        Y[bp + stride * lhs_k] = X[bp + stride * rhs_k];
        R1 = mask_mux_mod(R1 + P1, N2);
        R2 = mask_mux_mod(R2 + P2, N3);
        lhs_k++;
      }
    }
  }
}

// Three-factor prime factor algorithm
void prime_factor_3(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                    const fft_func_t *fs, int64_t bp, int64_t stride,
                    int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  int64_t N1 = Ns[0];
  int64_t N2 = Ns[1];
  int64_t N3 = Ns[2];
  int64_t N = N1 * N2 * N3;

  QValues B = compute_Qs(N1, N2, N3);

  int64_t Q1P = (-B.Q1) % N1;
  if (Q1P < 0)
    Q1P += N1;

  int64_t Q2P = (-B.Q2) % N2;
  if (Q2P < 0)
    Q2P += N2;

  int64_t P1 = (-B.Q4) % N2;
  if (P1 < 0)
    P1 += N2;

  int64_t P2 = ((-B.Q3) / N1) % N3;
  if (P2 < 0)
    P2 += N3;

  // Forward mapping
  nmap_3(Y, X, bp, stride, N1, N2, N3, Q1P, Q2P);

  // Create 3D arrays for FFT operations
  MDArray Y123 = create_mdarray(Y, Ns, 3);
  MDArray X123 = create_mdarray(X, Ns, 3);

  // Perform FFTs
  do_fft<fft_func_t>(&X123, &Y123, fs, Ns, 3, es, 0, bp, stride,
                     flags);
  do_fft<fft_func_t>(&Y123, &X123, fs, Ns, 3, es, 1, bp, stride,
                     flags);
  do_fft<fft_func_t>(&X123, &Y123, fs, Ns, 3, es, 2, bp, stride,
                     flags);
  // Rebind
  Y = Y123.data;
  X = X123.data;

  kmap_3(Y, X, bp, stride, N1, N2, N3, P1, P2);

  *YY = Y;
  *XX = X;
}

void pfa_extend_4(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                  const fft_func_t *fs, int64_t bp,
                  int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int64_t N1E = Ns[0] * Ns[1];
  const int64_t N2E = Ns[2] * Ns[3];
  const int64_t NsE[] = {N1E, N2E};
  const int64_t Ns1[] = {Ns[0], Ns[1]};
  const int64_t Ns2[] = {Ns[2], Ns[3]};

  ExtendedEuclidResult result = extended_euclid(N1E, N2E);
  minassert(result.g == 1, "prime_factor N1 and N2 must be coprime");

  int64_t M1 = result.x;
  int64_t M2 = result.y;

  int64_t Q1P = M2 % N1E;
  if (Q1P < 0)
    Q1P += N1E;

  nmap_2(Y, X, bp, stride, N1E, N2E, Q1P);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa2_t>(&X2D, &Y2D, fs, Ns1, 2, es, 0, bp, stride, flags);
  do_fft<pfa2_t>(&Y2D, &X2D, fs+2, Ns2, 2, es+2, 1, bp, stride, flags);
 
  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  // Backward mapping
  int64_t Q2P = M1 % N2E;
  if (Q2P < 0)
    Q2P += N2E;

  kmap_2(X, Y, bp, stride, N1E, N2E, Q2P);

  *YY = X;
  *XX = Y;
}

void pfa_extend_5(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                  const fft_func_t *fs, int64_t bp,
                  int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int64_t N1E = Ns[0] * Ns[1] * Ns[2];
  const int64_t N2E = Ns[3] * Ns[4];
  const int64_t NsE[] = {N1E, N2E};
  const int64_t Ns1[] = {Ns[0], Ns[1], Ns[2]};
  const int64_t Ns2[] = {Ns[3], Ns[4]};
  ExtendedEuclidResult result = extended_euclid(N1E, N2E);
  minassert(result.g == 1, "prime_factor N1 and N2 must be coprime");

  int64_t M1 = result.x;
  int64_t M2 = result.y;

  int64_t Q1P = M2 % N1E;
  if (Q1P < 0)
    Q1P += N1E;

  nmap_2(Y, X, bp, stride, N1E, N2E, Q1P);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa3_t>(&X2D, &Y2D, fs, Ns1, 2, es, 0, bp, stride, flags);
  do_fft<pfa2_t>(&Y2D, &X2D, fs+3, Ns2, 2, es+3, 1, bp, stride, flags);
 
  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  // Backward mapping
  int64_t Q2P = M1 % N2E;
  if (Q2P < 0)
    Q2P += N2E;

  kmap_2(X, Y, bp, stride, N1E, N2E, Q2P);

  *YY = X;
  *XX = Y;
}

void pfa_extend_6(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                  const fft_func_t *fs, int64_t bp,
                  int64_t stride, int32_t flags) {
  MFFTELEM *__restrict__ Y = *YY;
  MFFTELEM *__restrict__ X = *XX;

  const int64_t N1E = Ns[0] * Ns[1] * Ns[2];
  const int64_t N2E = Ns[3] * Ns[4] * Ns[5];
  const int64_t NsE[] = {N1E, N2E};
  const int64_t Ns1[] = {Ns[0], Ns[1], Ns[2]};
  const int64_t Ns2[] = {Ns[3], Ns[4], Ns[5]};
  ExtendedEuclidResult result = extended_euclid(N1E, N2E);
  minassert(result.g == 1, "prime_factor N1 and N2 must be coprime");

  int64_t M1 = result.x;
  int64_t M2 = result.y;

  int64_t Q1P = M2 % N1E;
  if (Q1P < 0)
    Q1P += N1E;

  nmap_2(Y, X, bp, stride, N1E, N2E, Q1P);

  MDArray Y2D = create_mdarray(Y, NsE, 2);
  MDArray X2D = create_mdarray(X, NsE, 2);

  do_fft<pfa3_t>(&X2D, &Y2D, fs, Ns1, 2, es, 0, bp, stride, flags);
  do_fft<pfa3_t>(&Y2D, &X2D, fs+3, Ns2, 2, es+3, 1, bp, stride, flags);
 
  // Rebind X and Y
  Y = Y2D.data;
  X = X2D.data;

  // Backward mapping
  int64_t Q2P = M1 % N2E;
  if (Q2P < 0)
    Q2P += N2E;

  kmap_2(X, Y, bp, stride, N1E, N2E, Q2P);

  *YY = X;
  *XX = Y;
}