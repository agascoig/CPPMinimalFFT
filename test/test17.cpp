
#include <fftw3.h>
#include <stdio.h>

#include <cfloat>
#include <cinttypes>
#include <cstdarg>
#include <ctime>
#include <iostream>
#include <new>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "CPPMinimalFFT.hpp"
#include "hmean.hpp"
#include "plan.hpp"

static const int NUM_TIMED_TESTS = 20;
static const int OVERSAMPLE_FACTOR = 40;

using hashmap_t = std::unordered_map<std::string, int>;
using normal_dist_t = std::normal_distribution<double>;

static HarmonicMean hm;

class random_normal {
 public:
  random_normal(int64_t seed, double mean, double stddev)
      : rng(seed), dist(mean, stddev) {}
  std::mt19937 rng;
  normal_dist_t dist;
  MFFTELEM get_rv() { return MFFTELEM(dist(rng), dist(rng)); }
  MinAlignedVector get_rv(size_t n) {
    MinAlignedVector v(n);
    for (int i = 0; i < n; ++i) {
      MFFTELEM e = get_rv();
      v[i] = e;
    }
    return v;
  }
};

fftw_plan create_fftw_plan(int n, MFFTELEM *in, MFFTELEM *out, int inverse) {
  if (inverse)
    return fftw_plan_dft_1d(n, (fftw_complex *)in, (fftw_complex *)out,
                            FFTW_BACKWARD, FFTW_ESTIMATE);
  else
    return fftw_plan_dft_1d(n, (fftw_complex *)in, (fftw_complex *)out,
                            FFTW_FORWARD, FFTW_ESTIMATE);
}

int64_t power_of(int64_t b, int64_t N) {
  int64_t count = 0;
  if (N <= 0) return 0;
  while (N % b == 0) {
    N /= b;
    count++;
    if (N == 1) return count;
  }
  return 0;
}

void print_v(const char *name, MFFTELEM *v, size_t n) {
  printf("%s = \n", name);
  for (size_t i = 0; i < n; ++i) {
    printf("  (%2.2f,%2.2f)\n", std::real(v[i]), std::imag(v[i]));
  }
  printf("\n");
}

double get_s_time(int64_t start, int64_t end) {
  int64_t interval = end - start;
  double elapsed_ns = (double)interval;
  return elapsed_ns * 1e-9;
}

MFFTELEM *test_fft_kernel(int64_t repeat_count, MFFTELEM *Y_ref, MFFTELEM *Y,
                          MFFTELEM *X_ref, MFFTELEM *X, MFFTELEM *copy_X,
                          fftw_plan P_ref, MinimalPlan *P, int64_t N, int bm,
                          double *t_ref_s, double *t_s, int32_t num_factors,
                          int64_t *Ns, fft_func_t *fns, int32_t *es,
                          void (*parent_fn)(void), int32_t inverse,
                          double *std_dev, const int64_t *params) {
  int64_t t_ref_start, t_ref_end;
  int64_t inner_start, inner_end;
  int64_t t_start, t_end;
  *t_ref_s = 0.0;
  *t_s = 0.0;
  bool inplace = (P != nullptr && P->bt_flags(P_INPLACE)) ? true : false;
  memcpy(X, copy_X, N * sizeof(MFFTELEM));
  memcpy(X_ref, copy_X, N * sizeof(MFFTELEM));
  t_ref_start = mingettime();
  fftw_execute(P_ref);
  t_ref_end = mingettime();
  if (get_s_time(t_ref_start, t_ref_end) < 10e-6)
    repeat_count *= OVERSAMPLE_FACTOR;  // oversample if less than 10 us
  memcpy(X_ref, copy_X, N * sizeof(MFFTELEM));
  int64_t n = 0;
  t_ref_start = mingettime();
  while (n++ < repeat_count) {
    inner_start = mingettime();
    fftw_execute(P_ref);
    if (inplace && (n != repeat_count))
      memcpy(X_ref, copy_X, N * sizeof(MFFTELEM));
    inner_end = mingettime();
  }
  t_ref_end = mingettime();
  double mu = 0;  // Welford's algorithm for running variance
  double M2 = 0;
  double delta, delta2;
  t_start = mingettime();
  n = 0;
  while (n++ < repeat_count) {
    inner_start = mingettime();
    if (P != NULL) {
      P->execute_plan(Y, X, 0, 0, 1);
      if (inplace && (n != repeat_count))
        memcpy(X, copy_X, N * sizeof(MFFTELEM));
    } else {
      if (parent_fn == nullptr) {
        ((fft_func_t)fns[0])(&Y, &X, N, es[0], 0, 1, inverse);
      } else if (num_factors <= MAX_FACTORS) {
        ((parent_fn_t)parent_fn)(&Y, &X, Ns, es, 0, 1, inverse, fns, params);
      } else {
        minassert(0, "Too many factors here.");
      }
      memcpy(X, copy_X,
             N * sizeof(MFFTELEM));  // inner routines do not copy input, so
                                     // must do it
    }
    inner_end = mingettime();
    double x = get_s_time(inner_start, inner_end);
    delta = x - mu;
    mu += delta / (n + 1);
    delta2 = x - mu;
    M2 += delta * delta2;
  }
  t_end = mingettime();
  *std_dev = sqrt(1.0 * M2 / (n - 1));
  if (bm) {
    *t_ref_s = get_s_time(t_ref_start, t_ref_end) / (double)repeat_count;
    *t_s = get_s_time(t_start, t_end) / (double)repeat_count;
  }
  return Y;
}

void print_fns(char *buf, fft_func_t *fns) {
  char fn[256];
  if (fns == NULL) return;
  for (int i = 0; i < MAX_FACTORS; ++i) {
    if (fns[i] == NULL) continue;
    for (int j = 0; j < sizeof(dispatch) / sizeof(dispatch[0]); ++j) {
      if (((fft_func_t)(fns[i])) == dispatch[j]) {
        snprintf(fn, sizeof(fn), "fftr%d ", j);
        strcat(buf, fn);
      }
    }
    if (fns[i] == (fft_func_t)bluestein) {
      snprintf(fn, sizeof(fn), "bluestein ");
      strcat(buf, fn);
    } else if (fns[i] == (fft_func_t)direct_dft) {
      snprintf(fn, sizeof(fn), "direct_dft ");
      strcat(buf, fn);
    }
  }
  if (strlen(buf)) buf[strlen(buf) - 1] = 0;  // remove last space
}

void print_result(const char *preamble, const char *name, int64_t N,
                  int num_factors, int64_t *Ns, int bm, double t_ref_s,
                  double t_s, fft_func_t *fns, double std_dev) {
  char timing_str[256];
  if (bm) {
    snprintf(timing_str, sizeof(timing_str), "time = %2.2es xFFTW = %2.2e", t_s,
             t_s / t_ref_s);
  } else {
    snprintf(timing_str, sizeof(timing_str), "untimed");
  }
  char fn_str[256];
  fn_str[0] = '\0';
  print_fns(fn_str, fns);
  char factors_str[256];
  factors_str[0] = '\0';
  snprintf(factors_str, sizeof(factors_str), "N=%" PRId64 " [", N);
  char comma = ',';
  for (int i = 0; i < num_factors; i++) {
    if (i == (num_factors - 1)) comma = '\0';
    snprintf(factors_str + strlen(factors_str), sizeof(factors_str),
             "%" PRId64 "%c", Ns[i], comma);
  }
  printf("%s %s %s %s] std_dev=%2.2es (%s)\n", preamble, name, timing_str,
         factors_str, std_dev, fn_str);
  fflush(stdout);
}

void test_fft(random_normal &RNG, const char *name, int bm, int inverse,
              int64_t N, int *pc, int *fc, int num_factors,
              void (*parent_fn)(void), MinimalPlan *P, int64_t *Ns,
              fft_func_t *fns, int32_t *es) {
  struct timespec t_ref_start, t_ref_end;
  struct timespec t_start, t_end;
  double t_ref_s = DBL_MAX, t_s = DBL_MAX;
  MinAlignedVector Y_ref(N);
  MinAlignedVector X(N);
  MinAlignedVector Y(N);
  MinAlignedVector X_ref = RNG.get_rv(N);
  MinAlignedVector copy_X = X_ref;
  bool inplace = (P != nullptr && P->bt_flags(P_INPLACE)) ? true : false;
  fftw_plan P_ref = create_fftw_plan(N, X_ref.data(), Y_ref.data(), inverse);
  int test_repeat = bm ? NUM_TIMED_TESTS : 1;
  double std_dev;
  int64_t params[MAX_PFA_PARAMS] = {0};
  if (num_factors > 1) generate_pfa_params(num_factors, Ns, params);
  MFFTELEM *Y_result = test_fft_kernel(
      test_repeat, Y_ref.data(), Y.data(), X_ref.data(), X.data(),
      copy_X.data(), P_ref, P, N, bm, &t_ref_s, &t_s, num_factors, Ns, fns, es,
      parent_fn, inverse, &std_dev, params);
  // reporting
  if (bm) hm_add(&hm, t_s / t_ref_s);
  if (inplace && (approx_cmp_v(X.data(), Y_ref.data(), N) ||
                  approx_cmp_v(Y_ref.data(), Y_result, N))) {
    print_result("Failed for inplace ", name, N, num_factors, Ns, bm, t_ref_s,
                 t_s, fns, std_dev);
    (*fc)++;
  } else if (approx_cmp_v(Y_ref.data(), Y_result, N)) {
    print_result("Failed for", name, N, num_factors, Ns, bm, t_ref_s, t_s, fns,
                 std_dev);
    (*fc)++;
  } else {
    (*pc)++;
    print_result("Passed for", name, N, num_factors, Ns, bm, t_ref_s, t_s, fns,
                 std_dev);
  }
  fftw_destroy_plan(P_ref);
}

void hex_dump(const void *ptr, size_t len) {
  const unsigned char *data = (const unsigned char *)ptr;
  for (size_t i = 0; i < len; ++i) {
    printf("%02X ", data[i]);
    if ((i + 1) % 16 == 0) printf("\n");
  }
  if (len % 16 != 0) printf("\n");
}

std::string gen_key(void (*parent_fn)(void), void *fns, int64_t *N_vals,
                    int32_t *es, char bm, int32_t inverse) {
  std::string key;
  key.append(reinterpret_cast<const char *>(&parent_fn), sizeof(void (*)()));
  key.append(reinterpret_cast<const char *>(fns), sizeof(void *) * MAX_FACTORS);
  key.append(reinterpret_cast<const char *>(N_vals),
             sizeof(int64_t) * MAX_FACTORS);
  key.append(reinterpret_cast<const char *>(es), sizeof(int32_t) * MAX_FACTORS);
  key.append(reinterpret_cast<const char *>(&bm), sizeof(char));
  key.append(reinterpret_cast<const char *>(&inverse), sizeof(inverse));
  return key;
}

int64_t prod(int64_t *arr, int len) {
  int64_t p = 1;
  for (int i = 0; i < len; ++i) {
    p *= arr[i];
  }
  return p;
}

void driver(random_normal &RNG, hashmap_t &d, int *radix, int radix_count,
            int num_factors, int bm, int *pc, int *fc, int32_t inverse,
            void (*parent_fn)(void), const int64_t *N_vals, int N_val_count,
            const char *name) {
  int64_t bs[MAX_FACTORS];
  int32_t es[MAX_FACTORS];
  fft_func_t fns[MAX_FACTORS];
  int64_t Ns[MAX_FACTORS];
  for (int t = 0; t < N_val_count; t++) {
    int64_t i = 0;
    memset(fns, 0, MAX_FACTORS * sizeof(fft_func_t));
    memset(Ns, 0, MAX_FACTORS * sizeof(int64_t));
    memset(es, 0, MAX_FACTORS * sizeof(int32_t));
    minassert(num_factors <= MAX_FACTORS, "Too many factors.");
    for (int f = 0; f < num_factors; f++) {
      const int64_t factor = N_vals[t * num_factors + f];
      Ns[f] = factor;
      for (int j = 0; j < radix_count; j++) {
        int r = radix[j];
        int64_t e = power_of(r, factor);
        if (e != 0) {
          es[i] = e;
          bs[i] = r;
          if ((r < DISPATCH_SZ) && (dispatch[r]))
            fns[i] = dispatch[r];
          else if (factor < DIRECT_SZ)
            fns[i] = &direct_dft;
          else
            fns[i] = &bluestein;
          i++;
          break;
        }
      }
      std::string key =
          gen_key((void (*)())(parent_fn), fns, Ns, es, bm, inverse);
      if (d.find(key) != d.end() || i != num_factors) continue;
      d[key] = 1;
      test_fft(RNG, name, bm, inverse, prod(Ns, num_factors), pc, fc,
               num_factors, parent_fn, 0, Ns, fns, es);
    }
  }
}

void print_time() {
  time_t now = time(NULL);
  struct tm *t = localtime(&now);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
  printf("# %s\n", buf);
}

void print_compiler_ver() {
#ifdef __clang__
  std::cout << "# Clang " << __clang_major__ << "." << __clang_minor__ << "."
            << __clang_patchlevel__;
#elif defined(__GNUC__)
  std::cout << "# GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "."
            << __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
  std::cout << "# MSVC " << _MSC_VER;
#else
  std::cout << "Unknown Compiler (" << __VERSION__ << ")";
#endif
  std::cout << std::endl << std::endl << std::flush;
}

static const int64_t factor_1[][1] = {
    {8}, {4}, {25}, {27}, {16}, {125}, {49}, {64}, {81}, {9 * 9 * 9}, {256}};

static const int64_t factor_2[][2] = {{4, 25},   {25, 4}, {4, 49},   {8, 9},
                                      {256, 25}, {16, 5}, {8, 7},    {11, 8},
                                      {49, 3},   {9, 8},  {25, 256}, {1, 256}};

static const int64_t factor_3[][3] = {
    {9, 5, 49},    {9, 49, 5},    {5, 9, 49},    {49, 5, 9},   {8, 7, 25},
    {7, 25, 8},    {2, 3, 5},     {2, 5, 3},     {3, 2, 5},    {3, 5, 2},
    {64, 3, 5},    {3, 5, 64},    {5, 64, 3},    {1, 1, 64},   {64, 1, 1},
    {27, 625, 49}, {625, 27, 49}, {49, 27, 625}, {49, 625, 27}};

static const int64_t high_radix_factor_1[][1] = {
    {262144}, {2097152},  // powers of 8
    {531441}, {4782969},  // powers of 9
    {78125},  {390625},   // powers of 5
    {823543}, {5764801}   // powers of 7
};

static const int64_t factor_4[][4] = {
    {3, 5, 7, 11},    {11, 7, 5, 3},  {25, 27, 49, 11}, {49, 27, 25, 11},
    {49, 25, 27, 11}, {7, 25, 9, 8},  {9, 25, 7, 8},    {8, 7, 25, 9},
    {81, 49, 25, 11}, {1, 25, 49, 81}};

static const int64_t factor_5[][5] = {{3, 5, 7, 11, 13},   {13, 11, 7, 5, 3},
                                      {25, 3, 49, 11, 13}, {49, 9, 25, 11, 1},
                                      {7, 25, 27, 11, 13}, {7, 25, 9, 2, 13},
                                      {9, 5, 7, 8, 13},    {8, 7, 5, 9, 13},
                                      {81, 49, 5, 17, 13}, {27, 5, 49, 11, 13}};

static const int64_t factor_6[][6] = {{4, 5, 7, 9, 11, 17}};

static const int64_t bluestein_1[][1] = {{15}, {16}, {11}, {13}, {17}};

void bluestein_test_parent(MFFTELEM **YY, MFFTELEM **XX, const int64_t *Ns,
                           const int32_t *es, const int64_t bp,
                           const int64_t stride, const int32_t flags,
                           const fft_func_t *fs, const int64_t *params) {
  bluestein(YY, XX, Ns[0], es[0], bp, stride, flags);
}

int main() {
  hashmap_t d;
  random_normal RNG(6502, 0.0, 1.0);
  hm_init(&hm);
  int pass = 0, fail = 0;
  int *pc = &pass;
  int *fc = &fail;
  print_time();
  print_compiler_ver();
  for (int n = 1; n < DIRECT_SZ; ++n) {
    int64_t N = n;
    fft_func_t fns[MAX_FACTORS] = {nullptr};
    fns[0] = &direct_dft;
    int32_t es = 1;
    test_fft(RNG, "direct_dft", 1, 0, N, pc, fc, 1, NULL, nullptr, &N, fns,
             &es);
  }

#define RUN_DRIVER(radix_arr, num_factors, bm, inverse, parent_fn, N_vals, \
                   name)                                                   \
  driver(RNG, d, (radix_arr), sizeof(radix_arr) / sizeof((radix_arr)[0]),  \
         (num_factors), 1 /*(bm)*/, pc, fc, (inverse),                     \
         (void (*)())(parent_fn), &(N_vals)[0][0],                         \
         sizeof((N_vals)) / sizeof(N_vals[0]), (name))

  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, 0, 0, nullptr, factor_1,
             "stockham test 0");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, 0, 0, nullptr, factor_1,
             "stockham test 1");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 1, 0, 0, nullptr, factor_1,
             "stockham test 2");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 1, 0, 0, nullptr, factor_1,
             "stockham test 3");
  RUN_DRIVER(((int[]){4, 9, 5, 7}), 1, 0, 0, nullptr, factor_1,
             "stockham test 4");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 1, 0, 0, nullptr, factor_1,
             "stockham test 5");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 1, 0, 0, nullptr, factor_1,
             "stockham test 6");
  d.clear();
  RUN_DRIVER(((int[]){2}), 1, 1, 0, nullptr, factor_1, "timed stockham test 0");
  RUN_DRIVER(((int[]){3}), 1, 1, 0, nullptr, factor_1, "timed stockham test 1");
  RUN_DRIVER(((int[]){4}), 1, 1, 0, nullptr, factor_1, "timed stockham test 2");
  RUN_DRIVER(((int[]){5}), 1, 1, 0, nullptr, factor_1, "timed stockham test 3");
  RUN_DRIVER(((int[]){7}), 1, 1, 0, nullptr, factor_1, "timed stockham test 4");
  RUN_DRIVER(((int[]){8}), 1, 1, 0, nullptr, factor_1, "timed stockham test 5");
  RUN_DRIVER(((int[]){9}), 1, 1, 0, nullptr, factor_1, "timed stockham test 6");
  d.clear();
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, 0, 1, nullptr, factor_1,
             "stockham inverse test 0");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, 0, 1, nullptr, factor_1,
             "stockham inverse test 1");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 1, 0, 1, nullptr, factor_1,
             "stockham inverse test 2");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 1, 0, 1, nullptr, factor_1,
             "stockham inverse test 3");
  RUN_DRIVER(((int[]){4, 9, 5, 7}), 1, 0, 1, nullptr, factor_1,
             "stockham inverse test 4");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 1, 0, 1, nullptr, factor_1,
             "stockham inverse test 5");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 1, 0, 1, nullptr, factor_1,
             "stockham inverse test 6");
  d.clear();
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 2, 0, 1, prime_factor_2, factor_2,
             "prime factor 2 test 0 inverse");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 2, 1, 0, prime_factor_2, factor_2,
             "prime factor 2 test 1 timed");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 2, 0, 0, prime_factor_2, factor_2,
             "prime factor 2 test 2");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 2, 0, 0, prime_factor_2, factor_2,
             "prime factor 2 test 3");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 2, 0, 0, prime_factor_2, factor_2,
             "prime factor 2 test 4");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 2, 0, 0, prime_factor_2, factor_2,
             "prime factor 2 test 5");
  d.clear();
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 3, 0, 1, prime_factor_3, factor_3,
             "prime factor 3 test 0 inverse");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 3, 1, 0, prime_factor_3, factor_3,
             "prime factor 3 test 1");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 3, 0, 0, prime_factor_3, factor_3,
             "prime factor 3 test 2");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 3, 0, 0, prime_factor_3, factor_3,
             "prime factor 3 test 3");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 3, 0, 0, prime_factor_3, factor_3,
             "prime factor 3 test 4");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 3, 0, 0, prime_factor_3, factor_3,
             "prime factor 3 test 5");
  d.clear();
  RUN_DRIVER(((int[]){2}), 1, 0, 0, nullptr, high_radix_factor_1,
             "radix 2 test");
  RUN_DRIVER(((int[]){3}), 1, 0, 0, nullptr, high_radix_factor_1,
             "radix 3 test");
  RUN_DRIVER(((int[]){4}), 1, 0, 0, nullptr, high_radix_factor_1,
             "radix 4 test");
  RUN_DRIVER(((int[]){5}), 1, 0, 0, nullptr, high_radix_factor_1,
             "radix 5 test");
  RUN_DRIVER(((int[]){7}), 1, 0, 0, nullptr, high_radix_factor_1,
             "radix 7 test");
  RUN_DRIVER(((int[]){8}), 1, 0, 0, nullptr, high_radix_factor_1,
             "radix 8 test");
  RUN_DRIVER(((int[]){9}), 1, 0, 0, nullptr, high_radix_factor_1,
             "radix 9 test");
  d.clear();
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 4, 0, 0, pfa_extend_4, factor_4,
             "prime factor extend 4");
  RUN_DRIVER(((int[]){2, 3, 5, 7, 11, 13, 17}), 5, 0, 0, pfa_extend_5, factor_5,
             "prime factor extend 5");
  RUN_DRIVER(((int[]){2, 3, 5, 7, 11, 13, 17}), 6, 0, 0, pfa_extend_6, factor_6,
             "prime factor extend 6");
  d.clear();
  RUN_DRIVER(((int[]){15, 16, 11, 13, 17}), 1, 0, 0, bluestein_test_parent,
             bluestein_1, "bluestein test");
  RUN_DRIVER(((int[]){15, 16, 11, 13, 17}), 1, 0, 1, bluestein_test_parent,
             bluestein_1, "bluestein inverse test");
  d.clear();
  static int64_t planner_n[] = {15,
                                100,
                                196,
                                72,
                                6400,
                                80,
                                56,
                                147,
                                72,
                                2205,
                                1400,
                                30,
                                960,
                                826875,
                                2 * 3 * 5 * 7 * 11 * 13,
                                8 * 25 * 7 * 3,
                                2 * 25 * 49 * 9,
                                1 << 20,
                                1 << 22};
  int64_t *planner_n_inverse = planner_n;
  for (int i = 0; i < sizeof(planner_n) / sizeof(planner_n[0]); ++i) {
    int64_t N = planner_n[i];
    int64_t *N_p = &N;
    const int32_t region = 0;
    MinimalPlan P(N_p, 1, region, region, P_NONE);
    test_fft(RNG, "planner", 1, 0, N, pc, fc, 1, NULL, &P, &N,
             P.get_funcs(region), NULL);
    MinimalPlan P_inv(N_p, 1, region, region, P_INVERSE);
    test_fft(RNG, "planner inverse", 1, 1, N, pc, fc, 1, NULL, &P_inv, &N,
             P_inv.get_funcs(region), NULL);
  }
  printf("\nPassed %d tests.\n", pass);
  printf("Failed %d tests.\n", fail);
  double hmv = hm_value(&hm);
  printf("Mean xFFTW = %2.2e\n", m_value(&hm));
  printf("Harmonic mean xFFTW = %2.2e\n", hmv);
  printf("Geometric mean xFFTW = %2.2e\n", gm_value(&hm));
  fflush(stdout);
  return 0;
}