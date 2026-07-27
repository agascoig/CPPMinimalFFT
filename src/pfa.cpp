
#include "pfa.hpp"

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

void QPs(int64_t* params, const int nf, const int64_t* Ns) noexcept {
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
  for (int i = 0; i < nf - 1; ++i) {
    N *= Ns[i];
    r = extended_euclid(Ns[i + 1], N);
    y = r.y % Ns[i + 1];
    if (y < 0) y += Ns[i + 1];
    params[nf - 1 + i] = y;
  }
}

// Inline branchless mod function
static inline int64_t mask_mux_mod(int64_t a, int64_t B) {
  return a - (B & -(a >= B));
}

template <int nf, typename T>
void nmap(T* __restrict__ Y, T* __restrict__ X, const int64_t bp,
          const int64_t stride, const int64_t* Ns, const int64_t* QP) noexcept {
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
void kmap(T* __restrict__ Y, T* __restrict__ X, const int64_t bp,
          const int64_t stride, const int64_t* Ns, const int64_t* QP) noexcept {
  int64_t buf[nf * 2 - 1] = {0};
  int64_t* np = &buf[0];
  int64_t* R = &buf[nf];
  int64_t k, N;
  int64_t lhs_k_stride = bp;
  int64_t np_p = nf - 1;
  while (1) {
    k = np[0];
    N = Ns[0];
    for (int i = 1; i < nf; ++i) {
      k += N * mask_mux_mod(np[i] + R[i - 1], Ns[i]);
      R[i - 1] = mask_mux_mod(R[i - 1] + QP[nf + i - 2], Ns[i]);
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

template <int NF>
void prime_factor(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd,
                  const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept {
  MFFTELEM* __restrict__ Y = *YY;
  MFFTELEM* __restrict__ X = *XX;
  if (rd.nm != nullptr) {
    int64_t lhs_stride = bp;
    int64_t rd_n = rd.n;
    const MAP_CACHE_T *nm_p = rd.nm;
    for (int64_t i = 0; i < rd_n; ++i) {
      int64_t n = nm_p[i];
      Y[lhs_stride] = X[bp + stride * n];
      lhs_stride += stride;
    }
  } else
    nmap<NF, MFFTELEM>(Y, X, bp, stride, rd.ns, rd.QPs);
  for (int i = 0; i < NF; ++i) {
    if ((i & 1) == 0)
      do_fft<NF>(XX, YY, rd, bp, stride, flags, i);  // input is in YY
    else
      do_fft<NF>(YY, XX, rd, bp, stride, flags, i);  // input is in XX
  }
  // put output to input in X for kmap
  if ((NF & 1) == 1) {
    Y = *YY;
    X = *XX;
  } else {
    Y = *XX;
    X = *YY;
  }
  if (rd.km != nullptr) {
    int64_t lhs_stride = bp;
    int64_t rd_n = rd.n;
    const MAP_CACHE_T *km_p = rd.km;
    for (int64_t i = 0; i < rd_n; ++i) {
      int64_t k = km_p[i];
      Y[lhs_stride] = X[bp + stride * k];
      lhs_stride += stride;
    }
  } else
    kmap<NF, MFFTELEM>(Y, X, bp, stride, rd.ns, rd.QPs);
  *YY = Y;
  *XX = X;
}

void prime_factor(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd,
                  const int64_t bp, const int64_t stride,
                  const int32_t flags, const int32_t nf) noexcept {
  // predictable branch targets
  switch (nf) {
    case 2:
      prime_factor<2>(YY, XX, rd, bp, stride, flags);
      break;
    case 3:
      prime_factor<3>(YY, XX, rd, bp, stride, flags);
      break;
    case 4:
      prime_factor<4>(YY, XX, rd, bp, stride, flags);
      break;
    case 5:
      prime_factor<5>(YY, XX, rd, bp, stride, flags);
      break;
    case 6:
      prime_factor<6>(YY, XX, rd, bp, stride, flags);
      break;
    case 7:
      prime_factor<7>(YY, XX, rd, bp, stride, flags);
      break;
    default:
      minassert(0, "One, or too many factors here.");
  }
  // *YY and *XX may have flipped
}

typedef void (*map_fn_t)(MAP_CACHE_T* __restrict__ Y,
                         MAP_CACHE_T* __restrict__ X, const int64_t bp,
                         const int64_t stride, const int64_t* Ns,
                         const int64_t* QPs) noexcept;

MAP_CACHE_T* generate_nmap(const int nf, const int64_t N, const int64_t* Ns,
                           const int64_t* QPs) noexcept {
  static const map_fn_t nmap_fn[] = {nullptr,
                                     nullptr,
                                     nmap<2, MAP_CACHE_T>,
                                     nmap<3, MAP_CACHE_T>,
                                     nmap<4, MAP_CACHE_T>,
                                     nmap<5, MAP_CACHE_T>,
                                     nmap<6, MAP_CACHE_T>,
                                     nmap<7, MAP_CACHE_T>};

  MAP_CACHE_T* Y = new MAP_CACHE_T[N];
  MAP_CACHE_T* X = new MAP_CACHE_T[N];
  for (int i = 0; i < N; ++i) {
    X[i] = i;
  }
  nmap_fn[nf](Y, X, 0, 1, Ns, QPs);
  delete[] X;
  return Y;
}

MAP_CACHE_T* generate_kmap(const int nf, const int64_t N, const int64_t* Ns,
                           const int64_t* QPs) noexcept {
  static const map_fn_t kmap_fn[] = {nullptr,
                                     nullptr,
                                     kmap<2, MAP_CACHE_T>,
                                     kmap<3, MAP_CACHE_T>,
                                     kmap<4, MAP_CACHE_T>,
                                     kmap<5, MAP_CACHE_T>,
                                     kmap<6, MAP_CACHE_T>,
                                     kmap<7, MAP_CACHE_T>};
  MAP_CACHE_T* Y = new MAP_CACHE_T[N];
  MAP_CACHE_T* X = new MAP_CACHE_T[N];
  for (int i = 0; i < N; ++i) X[i] = i;
  kmap_fn[nf](Y, X, 0, 1, Ns, QPs);
  delete[] X;
  return Y;
}

// explicit template instantiations
template void prime_factor<2>(MFFTELEM** YY, MFFTELEM** XX,
                              const region_data& rd, const int64_t bp,
                              const int64_t stride,
                              const int32_t flags) noexcept;

template void prime_factor<3>(MFFTELEM** YY, MFFTELEM** XX,
                              const region_data& rd, const int64_t bp,
                              const int64_t stride,
                              const int32_t flags) noexcept;

template void prime_factor<4>(MFFTELEM** YY, MFFTELEM** XX,
                              const region_data& rd, const int64_t bp,
                              const int64_t stride,
                              const int32_t flags) noexcept;

template void prime_factor<5>(MFFTELEM** YY, MFFTELEM** XX,
                              const region_data& rd, const int64_t bp,
                              const int64_t stride,
                              const int32_t flags) noexcept;

template void prime_factor<6>(MFFTELEM** YY, MFFTELEM** XX,
                              const region_data& rd, const int64_t bp,
                              const int64_t stride,
                              const int32_t flags) noexcept;

template void prime_factor<7>(MFFTELEM** YY, MFFTELEM** XX,
                              const region_data& rd, const int64_t bp,
                              const int64_t stride,
                              const int32_t flags) noexcept;
