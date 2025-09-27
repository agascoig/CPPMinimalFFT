
#ifndef CMINIMALFFT_H
#define CMINIMALFFT_H

#include <cmath>
#include <complex>
#include <cstdlib>
#include <execinfo.h>
#include <iostream>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>

// Define the element type
typedef std::complex<double> MFFTELEM;

void minsincos(double x, double *s, double *c);

inline MFFTELEM minsincos(double angle) {
  double zi, zr;
  minsincos(angle, &zi, &zr); // calling openlibm
  return std::complex<double>(zr, zi);
}

static inline std::complex<double> times_pmim(std::complex<double> z,
                                              int inverse) {
  if (inverse) {
    return std::complex<double>(-std::imag(z), std::real(z));
  } else {
    return std::complex<double>(std::imag(z), -std::real(z));
  }
}

static void swap_ptrs(MFFTELEM **a, MFFTELEM **b) {
  MFFTELEM *temp = *a;
  *a = *b;
  *b = temp;
}

static void print_stacktrace(void) {
  void *buffer[100];
  int nptrs = backtrace(buffer, 100);
  char **strings = backtrace_symbols(buffer, nptrs);
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

#define minassert(cond, msg)                                                   \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "Assertion failed: %s\n", msg);                          \
      print_stacktrace();                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T> class MinAlignedAllocator {
public:
  using value_type = T;

  MinAlignedAllocator() noexcept = default;
  template <typename U>
  MinAlignedAllocator(const MinAlignedAllocator<U> &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t n) {
    // Calculate the size needed for n elements

    std::size_t size = n * sizeof(T);
    // Allocate memory with alignment of T
    void *ptr =
        std::aligned_alloc(sizeof(T), size); // align on complex boundary
    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, std::size_t) noexcept { std::free(p); }
};

template <typename T, typename U>
bool operator==(const MinAlignedAllocator<T> &,
                const MinAlignedAllocator<U> &) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(const MinAlignedAllocator<T> &,
                const MinAlignedAllocator<U> &) noexcept {
  return false;
}
using MinAlignedVector = std::vector<MFFTELEM, MinAlignedAllocator<MFFTELEM>>;

static inline void *minaligned_alloc(size_t alignment, size_t sz,
                                     size_t count) {
  void *p = aligned_alloc(alignment, sz * count);
  minassert(p, "Memory allocation failed.");
  return p;
}

static inline void *minaligned_calloc(size_t alignment, size_t sz,
                                      size_t count) {
  void *p = aligned_alloc(alignment, sz * count);
  if (!p) {
    print_stacktrace();
  }
  minassert(p, "Memory allocation failed.");
  memset(p, 0, sz * count);
  return p;
}

static int approx_cmp(MFFTELEM x, MFFTELEM y) {
  // borrowed from Julia: rtol = sqrt(eps(eltype(x)))
  double atol = 0;
  double rtol = sizeof(MFFTELEM) == 2 * 8   ? 1.4901161193847656e-8
                : sizeof(MFFTELEM) == 2 * 4 ? 0.00034526698
                                            : 0.03125;

  if (x == y)
    return 0;

  // Check for finite values
  if (isfinite(std::real(x)) && isfinite(std::imag(x)) &&
      isfinite(std::real(y)) && isfinite(std::imag(y))) {
    double diff = std::abs(x - y);
    double norm_x = std::abs(x);
    double norm_y = std::abs(y);
    double tol = std::fmax(atol, rtol * std::fmax(norm_x, norm_y));
    if (diff <= tol)
      return 0;
  }

  return 1;
}

static double norm_v(MFFTELEM *x, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double zr = std::real(x[i]);
    double zi = std::imag(x[i]);
    sum += zr * zr + zi * zi;
  }
  return sqrt(sum);
}

static int is_finite(MFFTELEM *x, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!isfinite(std::real(x[i])) || !isfinite(std::imag(x[i]))) {
      return 0;
    }
  }
  return 1;
}

static int approx_cmp_v(MFFTELEM *x, MFFTELEM *y, size_t n) {
  double atol = 0;
  double rtol = sizeof(MFFTELEM) == 2 * 8   ? 1.4901161193847656e-8
                : sizeof(MFFTELEM) == 2 * 4 ? 0.00034526698
                                            : 0.03125;

  if (x == y)
    return 0;

    double diff = 0.0;

  // Check for finite values
  if (is_finite(x, n) && is_finite(y, n)) {
    for (int i = 0; i < n; ++i) {
      double zr = std::real(x[i]-y[i]);
      double zi = std::imag(x[i]-y[i]);

      diff += zr * zr + zi * zi;
    }
    diff = sqrt(diff);
    double norm_x = norm_v(x, n);
    double norm_y = norm_v(y, n);
    double tol = std::fmax(atol, rtol * std::fmax(norm_x, norm_y));
    if (diff <= tol)
      return 0;
  }

//  std::cout << "diff error: " << diff << std::endl;
  return 1;
}

#define MAX_DIMS 8
#define MAX_REGIONS MAX_DIMS

// Structure to represent multi-dimensional or decomposed array info
typedef struct {
  MFFTELEM *data;
  int64_t dims[MAX_DIMS];
  int64_t total_size;
  int32_t ndims;
} MDArray;

static MDArray create_mdarray(MFFTELEM *data, const int64_t *dims, int32_t ndims) {
  MDArray arr;
  arr.data = data;
  memcpy(arr.dims, dims, ndims * sizeof(int64_t));
  arr.ndims = ndims;

  arr.total_size = 1;
  for (int64_t i = 0; i < ndims; i++) {
    arr.total_size *= dims[i];
  }

  return arr;
}

static void free_mdarray(MDArray *arr) {
  free(arr); // do not free the data itself
}

// fft_func_t: tag for do_fft
typedef void (*fft_func_t)(MFFTELEM **Y, MFFTELEM **X, int64_t N, int32_t e1,
                           int64_t bp, int64_t stride, int32_t flags);

typedef struct MinimalPlan MinimalPlan;

void do_fft_planned(MinimalPlan *P, MDArray *oy, MDArray *ix, int64_t r);

template <typename Func>
void do_fft(MDArray *oy, MDArray *ix, const fft_func_t *fs,
            const int64_t *Ns, int64_t ndims, const int32_t *es, int64_t r, int64_t bp,
            int64_t stride, int32_t flags);

void fftr2(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t flags);
void fftr3(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t flags);
void fftr4(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t flags);
void fftr5(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t flags);
void fftr7(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t flags);
void fftr8(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t flags);
void fftr9(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t flags);
void direct_dft(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
                int64_t stride, int32_t flags);
void bluestein(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
               int64_t stride, int32_t flags);

static fft_func_t dispatch[] = {NULL,   NULL, &fftr2, &fftr3, &fftr4,
                                &fftr5, NULL, &fftr7, &fftr8, &fftr9};
static const int dispatch_sz = sizeof(dispatch) / sizeof(dispatch[0]);

void prime_factor_2(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                    const fft_func_t *fs, int64_t bp, int64_t stride,
                    int32_t flags);

typedef struct {} pfa2_t; // tag for do_fft

void prime_factor_3(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                    const fft_func_t *fs, int64_t bp, int64_t stride,
                    int32_t flags);

typedef struct {} pfa3_t; // tag for do_fft

typedef void (*parent_fn_t)(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                    const fft_func_t *fs, int64_t bp, int64_t stride,
                    int32_t flags); // for casts

void pfa_extend_4(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                  const fft_func_t *fs, int64_t bp,
                  int64_t stride, int32_t flags);

void pfa_extend_5(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                  const fft_func_t *fs, int64_t bp,
                  int64_t stride, int32_t flags);

void pfa_extend_6(MFFTELEM **YY, MFFTELEM **XX, const int32_t *es, const int64_t *Ns,
                  const fft_func_t *fs, int64_t bp,
                  int64_t stride, int32_t flags);

static inline int64_t count_leading_zeros(uint64_t x) {
  return __builtin_clzll(x);
}

#endif // CMINIMALFFT_H