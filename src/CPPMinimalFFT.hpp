
#ifndef __CMINIMALFFT_H__
#define __CMINIMALFFT_H__

#include <execinfo.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#if defined(MINFFT) && (MINFFT == 32)
typedef float MFFTELEMRI;
#else
typedef double MFFTELEMRI;
#endif

typedef std::complex<MFFTELEMRI> MFFTELEM;

#if !defined(__APPLE__)
// linux
static inline uint64_t clock_gettime_nsec_np(clockid_t clk_id) {
  struct timespec ts;
  clock_gettime(clk_id, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
#endif

static inline uint64_t mingettime() {
  return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
}

/*
// TBD: remove
static inline std::complex<MFFTELEMRI> times_im(std::complex<MFFTELEMRI> z) {
  return std::complex<MFFTELEMRI>(-std::imag(z), std::real(z));
}

// TBD: remove
static inline std::complex<MFFTELEMRI> times_nim(std::complex<MFFTELEMRI> z) {
  return std::complex<MFFTELEMRI>(std::imag(z), -std::real(z));
}
*/

template <typename T>
inline T MUL_I_CPX(T x) {
  return T(-std::imag(x), std::real(x));
}

template <typename T>
inline T MUL_NI_CPX(T x) {
  return T(std::imag(x), -std::real(x));
}

template <typename T>
inline T MUL_IM_CPX(MFFTELEMRI a, T x) {
  return a * MUL_I_CPX(x);
}

template <typename T>
inline T MUL_NIM_CPX(MFFTELEMRI a, T x) {
  return a * MUL_NI_CPX(x);
}

// (a+ ia)*(b+ic)
template <typename T>
inline T MUL_RI_CPX(MFFTELEMRI a, T x) {
  MFFTELEMRI b = std::real(x);
  MFFTELEMRI c = std::imag(x);
  return T(a * (b - c), a * (b + c));
}

static void print_stacktrace(void) {
  void* buffer[100];
  int nptrs = backtrace(buffer, 100);
  char** strings = backtrace_symbols(buffer, nptrs);
  if (strings == NULL) {
    perror("backtrace_symbols");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Stacktrace:\n");
  for (int i = 0; i < nptrs; i++) {
    fprintf(stderr, "%s\n", strings[i]);
  }
  free(strings);
}

#ifdef NDEBUG
#define minassert(cond, msg) ((void)0)
#else
#define minassert(cond, msg)                          \
  do {                                                \
    if (!(cond)) {                                    \
      fprintf(stderr, "Assertion failed: %s\n", msg); \
      print_stacktrace();                             \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while (0)
#endif

static const int ALIGN_SZ = 16;  // 8 for single-precision will not work

template <typename T>
class MinAlignedAllocator {
 public:
  using value_type = T;
  MinAlignedAllocator() noexcept = default;

  template <typename U>
  MinAlignedAllocator(const MinAlignedAllocator<U>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    // Calculate the size needed for n elements
    std::size_t size = n * sizeof(T);
    // Allocate memory with alignment of T
    //    void *ptr =
    //        aligned_alloc(sizeof(T), size);
    void* ptr = std::aligned_alloc(sizeof(T), size);  // simd: 16 byte alignment
    // std::cerr << "Alloc: " << ptr << std::endl;
    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) noexcept {
    // std::cerr << "Free: " << p << std::endl;
    std::free(p);
  }
};

template <typename T, typename U>
bool operator==(const MinAlignedAllocator<T>&,
                const MinAlignedAllocator<U>&) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(const MinAlignedAllocator<T>&,
                const MinAlignedAllocator<U>&) noexcept {
  return false;
}

using MinAlignedVector = std::vector<MFFTELEM, MinAlignedAllocator<MFFTELEM>>;

static inline void* minaligned_alloc(size_t alignment, size_t sz,
                                     size_t count) {
  void* p = aligned_alloc(alignment, sz * count);
  minassert(p, "Memory allocation failed.");
  return p;
}

static inline void* minaligned_calloc(size_t alignment, size_t sz,
                                      size_t count) {
  void* p = aligned_alloc(alignment, sz * count);
  if (!p) {
    print_stacktrace();
  }
  minassert(p, "Memory allocation failed.");
  memset(p, 0, sz * count);
  return p;
}

static int64_t prod(int nf, const int64_t* Ns) {
  int64_t N = 1;
  for (int i = 0; i < nf; ++i) N *= Ns[i];
  return N;
}

inline bool minisfinite(MFFTELEM x, MFFTELEM y) {
#if defined(__FINITE_MATH_ONLY__)
return true;
#else
  return std::isfinite(std::real(x)) && std::isfinite(std::imag(x)) &&
         std::isfinite(std::real(y)) && std::isfinite(std::imag(y));
#endif
}
static int approx_cmp(MFFTELEM x, MFFTELEM y) {
  // borrowed from Julia: rtol = sqrt(eps(eltype(x)))
  double atol = 0;
  double rtol = sizeof(MFFTELEM) == 2 * 8   ? 1.4901161193847656e-8
                : sizeof(MFFTELEM) == 2 * 4 ? 0.00034526698
                                            : 0.03125;

  if (x == y) return 0;
  // Check for finite values
  if (minisfinite(x,y)) {
    double diff = std::abs(x - y);
    double norm_x = std::abs(x);
    double norm_y = std::abs(y);
    double tol = std::fmax(atol, rtol * std::fmax(norm_x, norm_y));
    if (diff <= tol) return 0;
  }
  return 1;
}

static double norm_v(const MinAlignedVector& X, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double zr = std::real(X[i]);
    double zi = std::imag(X[i]);
    sum += zr * zr + zi * zi;
  }
  return sqrt(sum);
}

static int is_finite(const MinAlignedVector& X, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!minisfinite(X[i], X[i])) {
      //      std::cerr << "not finite at n=" << i << std::endl;
      return 0;
    }
  }
  return 1;
}

static int approx_cmp_v(const MinAlignedVector& X, const MinAlignedVector& Y,
                        size_t n) {
  double atol = 0;
  double rtol = sizeof(MFFTELEM) == 2 * 8   ? 1.4901161193847656e-8
                : sizeof(MFFTELEM) == 2 * 4 ? 0.00034526698
                                            : 0.03125;
  if (X == Y) return 0;
  double diff = 0.0;
  double tol = 0.0;
  // Check for finite values
  if (is_finite(X, n) && is_finite(Y, n)) {
    for (int i = 0; i < n; ++i) {
      double zr = std::real(X[i] - Y[i]);
      double zi = std::imag(X[i] - Y[i]);
      diff += zr * zr + zi * zi;
    }
    diff = sqrt(diff);
    double norm_x = norm_v(X, n);
    double norm_y = norm_v(Y, n);
    tol = std::fmax(atol, rtol * std::fmax(norm_x, norm_y));
    if (diff <= tol) return 0;
  }
  return 1;
}

inline constexpr int MAX_FACTORS = 7;
inline constexpr int MAX_DIMS = 8;
inline constexpr int MAX_REGIONS = MAX_DIMS;
inline constexpr int DEFAULT_DIRECT_SZ = 15;
inline constexpr int DEFAULT_SMALL_SZ = 28;
inline constexpr int MAX_PFA_PARAMS = (2 * (MAX_FACTORS - 1));
inline constexpr int MAX_MAP_CACHE = 1 << 12;
using MAP_CACHE_T = uint16_t;

// fft_func_t: tag for do_fft
typedef void (*fft_func_t)(MFFTELEM** Y, MFFTELEM** X, const int64_t N,
                           const int32_t e1, const int64_t bp,
                           const int64_t stride, const int32_t flags) noexcept;

struct region_data {
  int64_t n = 0;  // size of this region only
  int64_t ns[MAX_FACTORS] = {0};
  int64_t strides[MAX_FACTORS] = {0};
  int64_t QPs[MAX_PFA_PARAMS];
  const MAP_CACHE_T* nm = nullptr;
  const MAP_CACHE_T* km = nullptr;
  fft_func_t func[MAX_FACTORS] = {nullptr};
  int64_t base[MAX_FACTORS] = {0};
  int32_t exp[MAX_FACTORS] = {0};
  int32_t num_factors = 0;
};

static inline region_data create_region_data_strides(region_data& rd) {
  int64_t total_size = 1;
  int64_t* strides_p = rd.strides;
  int64_t* ns_p = rd.ns;
  int32_t num_factors = rd.num_factors;
  for (int64_t i = 0; i < num_factors; i++) {
    int64_t d = *ns_p++;
    *strides_p++ = total_size;
    total_size *= d;
  }
  rd.n = total_size;
  return rd;
}

static inline region_data create_region_data(
    const int64_t* __restrict__ dims, int32_t ndims, const fft_func_t* fns,
    const int64_t* base, const int32_t* exp, const int64_t* __restrict__ QPs,
    const MAP_CACHE_T* nm, const MAP_CACHE_T* km) {
  region_data rd;
  rd.num_factors = ndims;
  for (int i = 0; i < ndims; ++i) {
    rd.ns[i] = dims[i];
  }
  create_region_data_strides(rd);
  for (int i = 0; i < MAX_FACTORS; ++i) {
    rd.func[i] = fns[i];  // func determines base
    rd.base[i] = base[i];
    rd.exp[i] = exp[i];
  }
  for (int i = 0; i < MAX_PFA_PARAMS; ++i) {
    rd.QPs[i] = QPs[i];
  }
  rd.nm = nm;
  rd.km = km;
  return rd;
}

class MinimalPlan;

void do_fft_planned(const MinimalPlan& P, MFFTELEM** YY, MFFTELEM** XX,
                    int32_t r) noexcept;

template <int NF>
void do_fft(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd,
            const int64_t bp, const int64_t stride, const int32_t flags,
            const int32_t r) noexcept;

template <bool Inverse>
void fftr2(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride,
           const int32_t flags) noexcept;
template <>
void fftr2<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;
template <>
void fftr2<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                 const int32_t e1, const int64_t bp, const int64_t stride,
                 const int32_t flags) noexcept;

template <bool Inverse>
void fftr3(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride,
           const int32_t flags) noexcept;
template <>
void fftr3<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;
template <>
void fftr3<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                 const int32_t e1, const int64_t bp, const int64_t stride,
                 const int32_t flags) noexcept;

template <bool Inverse>
void fftr4(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride,
           const int32_t flags) noexcept;
template <>
void fftr4<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;
template <>
void fftr4<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                 const int32_t e1, const int64_t bp, const int64_t stride,
                 const int32_t flags) noexcept;

template <bool Inverse>
void fftr5(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride,
           const int32_t flags) noexcept;
template <>
void fftr5<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;
template <>
void fftr5<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                 const int32_t e1, const int64_t bp, const int64_t stride,
                 const int32_t flags) noexcept;

template <bool Inverse>
void fftr7(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride,
           const int32_t flags) noexcept;
template <>
void fftr7<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;
template <>
void fftr7<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                 const int32_t e1, const int64_t bp, const int64_t stride,
                 const int32_t flags) noexcept;

template <bool Inverse>
void fftr8(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride,
           const int32_t flags) noexcept;
template <>
void fftr8<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;
template <>
void fftr8<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                 const int32_t e1, const int64_t bp, const int64_t stride,
                 const int32_t flags) noexcept;

template <bool Inverse>
void fftr9(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
           const int64_t bp, const int64_t stride,
           const int32_t flags) noexcept;
template <>
void fftr9<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;
template <>
void fftr9<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                 const int32_t e1, const int64_t bp, const int64_t stride,
                 const int32_t flags) noexcept;

template <bool Inverse>
void fftr11(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
            const int64_t bp, const int64_t stride,
            const int32_t flags) noexcept;
template <>
void fftr11<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;
template <>
void fftr11<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;

template <bool Inverse>
void fftr13(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
            const int64_t bp, const int64_t stride,
            const int32_t flags) noexcept;
template <>
void fftr13<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;
template <>
void fftr13<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;

template <bool Inverse>
void fftr16(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
            const int64_t bp, const int64_t stride,
            const int32_t flags) noexcept;
template <>
void fftr16<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;
template <>
void fftr16<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;

template <bool Inverse>
void fftr17(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
            const int64_t bp, const int64_t stride,
            const int32_t flags) noexcept;
template <>
void fftr17<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;
template <>
void fftr17<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;

template <bool Inverse>
void fftr19(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
            const int64_t bp, const int64_t stride,
            const int32_t flags) noexcept;
template <>
void fftr19<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;
template <>
void fftr19<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;

template <bool Inverse>
void fftr23(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
            const int64_t bp, const int64_t stride,
            const int32_t flags) noexcept;
template <>
void fftr23<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;
template <>
void fftr23<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;

template <bool Inverse>
void fftr29(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
            const int64_t bp, const int64_t stride,
            const int32_t flags) noexcept;
template <>
void fftr29<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;
template <>
void fftr29<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;

template <bool Inverse>
void fftr31(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
            const int64_t bp, const int64_t stride,
            const int32_t flags) noexcept;
template <>
void fftr31<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;
template <>
void fftr31<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                  const int32_t e1, const int64_t bp, const int64_t stride,
                  const int32_t flags) noexcept;

template <bool Inverse>
void direct_dft(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
                const int64_t bp, const int64_t stride,
                const int32_t flags) noexcept;
template <bool Inverse>
void bluestein(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
               const int64_t bp, const int64_t stride,
               const int32_t flags) noexcept;

template <bool Inverse>
void small_1(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_1<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_1<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_2(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_2<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_2<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_3(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_3<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_3<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_4(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_4<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_4<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_5(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_5<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_5<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_6(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_6<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_6<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_7(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_7<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_7<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_8(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_8<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_8<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_9(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
             const int64_t bp, const int64_t stride,
             const int32_t flags) noexcept;
template <>
void small_9<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;
template <>
void small_9<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                   const int32_t e1, const int64_t bp, const int64_t stride,
                   const int32_t flags) noexcept;

template <bool Inverse>
void small_10(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_10<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_10<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_11(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_11<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_11<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_12(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_12<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_12<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_13(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_13<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_13<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_14(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_14<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_14<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_15(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_15<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_15<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_16(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_16<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_16<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_17(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_17<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_17<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_18(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_18<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_18<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_19(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_19<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_19<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_20(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_20<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_20<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_21(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_21<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_21<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_22(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_22<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_22<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_23(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_23<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_23<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_24(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_24<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_24<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_26(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_26<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_26<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

template <bool Inverse>
void small_28(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int32_t e1,
              const int64_t bp, const int64_t stride,
              const int32_t flags) noexcept;
template <>
void small_28<false>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                     const int32_t e1, const int64_t bp, const int64_t stride,
                     const int32_t flags) noexcept;
template <>
void small_28<true>(MFFTELEM** YY, MFFTELEM** XX, const int64_t N,
                    const int32_t e1, const int64_t bp, const int64_t stride,
                    const int32_t flags) noexcept;

bool small_available(const int64_t N, const bool Inverse);

fft_func_t get_small_func(const int64_t N, const bool Inverse);

static const int SMALL_SZ = 28;

static const fft_func_t dispatch[] = {
    nullptr,        nullptr,        &fftr2<false>, &fftr3<false>,
    &fftr4<false>,  &fftr5<false>,  nullptr,       &fftr7<false>,
    &fftr8<false>,  &fftr9<false>,  nullptr,       &fftr11<false>,
    nullptr,        &fftr13<false>, nullptr,       nullptr,
    &fftr16<false>, &fftr17<false>, nullptr,       &fftr19<false>,
    nullptr,        nullptr,        nullptr,       &fftr23<false>,
    nullptr,        nullptr,        nullptr,       nullptr,
    nullptr,        &fftr29<false>, nullptr,       &fftr31<false>};
static const fft_func_t dispatch_inverse[] = {
    nullptr,      nullptr,       &fftr2<true>,  &fftr3<true>,  &fftr4<true>,
    &fftr5<true>, nullptr,       &fftr7<true>,  &fftr8<true>,  &fftr9<true>,
    nullptr,      &fftr11<true>, nullptr,       &fftr13<true>, nullptr,
    nullptr,      &fftr16<true>, &fftr17<true>, nullptr,       &fftr19<true>,
    nullptr,      nullptr,       nullptr,       &fftr23<true>, nullptr,
    nullptr,      nullptr,       nullptr,       nullptr,       &fftr29<true>,
    nullptr,      &fftr31<true>};

static const int DISPATCH_SZ = sizeof(dispatch) / sizeof(dispatch[0]);

static inline int64_t count_leading_zeros(uint64_t x) {
  return __builtin_clzll(x);
}

struct fn_name_s {
  fft_func_t fn;
  const char* name;
} const fns_names[] = {
    {&bluestein<false>, "bluestein"},   {&bluestein<true>, "ibluestein"},
    {&direct_dft<false>, "direct_dft"}, {&direct_dft<true>, "idirect_dft"},
    {&fftr2<false>, "fftr2"},           {&fftr2<true>, "ifftr2"},
    {&fftr3<false>, "fftr3"},           {&fftr3<true>, "ifftr3"},
    {&fftr4<false>, "fftr4"},           {&fftr4<true>, "ifftr4"},
    {&fftr5<false>, "fftr5"},           {&fftr5<true>, "ifftr5"},
    {&fftr7<false>, "fftr7"},           {&fftr7<true>, "ifftr7"},
    {&fftr8<false>, "fftr8"},           {&fftr8<true>, "ifftr8"},
    {&fftr9<false>, "fftr9"},           {&fftr9<true>, "ifftr9"},
    {&fftr11<false>, "fftr11"},         {&fftr11<true>, "fftr11i"},
    {&fftr13<false>, "fftr13"},         {&fftr13<true>, "fftr13i"},
    {&fftr16<false>, "fftr16"},         {&fftr16<true>, "fftr16i"},
    {&fftr17<false>, "fftr17"},         {&fftr17<true>, "fftr17i"},
    {&fftr19<false>, "fftr19"},         {&fftr19<true>, "fftr19i"},
    {&fftr23<false>, "fftr23"},         {&fftr23<true>, "fftr23i"},
    {&fftr29<false>, "fftr29"},         {&fftr29<true>, "fftr29i"},
    {&fftr31<false>, "fftr31"},         {&fftr31<true>, "fftr31i"},
    {&small_1<false>, "small1"},        {&small_2<false>, "small2"},
    {&small_3<false>, "small3"},        {&small_4<false>, "small4"},
    {&small_5<false>, "small5"},        {&small_6<false>, "small6"},
    {&small_7<false>, "small7"},        {&small_8<false>, "small8"},
    {&small_9<false>, "small9"},        {&small_10<false>, "small10"},
    {&small_11<false>, "small11"},      {&small_12<false>, "small12"},
    {&small_13<false>, "small13"},      {&small_14<false>, "small14"},
    {&small_15<false>, "small15"},      {&small_16<false>, "small16"},
    {&small_17<false>, "small17"},      {&small_18<false>, "small18"},
    {&small_19<false>, "small19"},      {&small_20<false>, "small20"},
    {&small_21<false>, "small21"},      {&small_22<false>, "small22"},
    {&small_23<false>, "small23"},      {&small_24<false>, "small24"},
    {&small_26<false>, "small26"},      {&small_28<false>, "small28"},
    {&small_1<true>, "small1i"},        {&small_2<true>, "small2i"},
    {&small_3<true>, "small3i"},        {&small_4<true>, "small4i"},
    {&small_5<true>, "small5i"},        {&small_6<true>, "small6i"},
    {&small_7<true>, "small7i"},        {&small_8<true>, "small8i"},
    {&small_9<true>, "small9i"},        {&small_10<true>, "small10i"},
    {&small_11<true>, "small11i"},      {&small_12<true>, "small12i"},
    {&small_13<true>, "small13i"},      {&small_14<true>, "small14i"},
    {&small_15<true>, "small15i"},      {&small_16<true>, "small16i"},
    {&small_17<true>, "small17i"},      {&small_18<true>, "small18i"},
    {&small_19<true>, "small19i"},      {&small_20<true>, "small20i"},
    {&small_21<true>, "small21i"},      {&small_22<true>, "small22i"},
    {&small_23<true>, "small23i"},      {&small_24<true>, "small24i"},
    {&small_26<true>, "small26i"},      {&small_28<true>, "small28i"}};

inline std::ostream& operator<<(std::ostream& os, const region_data& rd) {
  const int64_t* n_p = rd.ns;
  const fft_func_t* f_p = rd.func;
  const int32_t* e_p = rd.exp;

  for (int32_t f = 0; f < rd.num_factors; f++) {
    int base = round(exp(log(n_p[f]) / e_p[f]));
    os << "    Factor " << f << ": base=" << base << " exp=" << e_p[f]
       << " ns=" << n_p[f] << " stride=" << rd.strides[f];
    fft_func_t func = f_p[f];
    for (int j = 0; j < sizeof(fns_names) / sizeof(fn_name_s); ++j) {
      if (func == fns_names[j].fn) {
        os << " func = " << fns_names[j].name;
        break;
      }
    }
    os << "\n";
  }
  return os;
}

#endif  // __CMINIMALFFT_H__