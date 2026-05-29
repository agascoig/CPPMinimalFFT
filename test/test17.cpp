
#include <cxxabi.h>
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
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <cfenv> // for NaN trapping
#include <buildinfo.h>

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
  random_normal(int64_t seed, double mean, double stddev) : rng(seed), dist(mean, stddev) {}
  std::mt19937 rng;
  normal_dist_t dist;
  MFFTELEM get_rand() { return MFFTELEM(dist(rng), dist(rng)); }
  MinAlignedVector get_rv(size_t n) {
    MinAlignedVector v(n);
    for (int i = 0; i < n; ++i) {
      MFFTELEM e = get_rand();
      // MFFTELEM e = MFFTELEM((double)i, 0.0);
      v[i] = e;
    }
    return v;
  }
};

fftwf_plan create_fftw_plan(int n, std::complex<float>* in, std::complex<float>* out, int inverse) {
  if (inverse)
    return fftwf_plan_dft_1d(n, (fftwf_complex*)in, (fftwf_complex*)out, FFTW_BACKWARD,
                             FFTW_ESTIMATE);
  else
    return fftwf_plan_dft_1d(n, (fftwf_complex*)in, (fftwf_complex*)out, FFTW_FORWARD,
                             FFTW_ESTIMATE);
}

fftw_plan create_fftw_plan(int n, std::complex<double>* in, std::complex<double>* out,
                           int inverse) {
  if (inverse)
    return fftw_plan_dft_1d(n, (fftw_complex*)in, (fftw_complex*)out, FFTW_BACKWARD, FFTW_ESTIMATE);
  else
    return fftw_plan_dft_1d(n, (fftw_complex*)in, (fftw_complex*)out, FFTW_FORWARD, FFTW_ESTIMATE);
}

void execute_fftw_plan(fftw_plan& P) { fftw_execute(P); }

void execute_fftw_plan(fftwf_plan& P) { fftwf_execute(P); }

void destroy_fftw_plan(fftw_plan& P) { fftw_destroy_plan(P); }

void destroy_fftw_plan(fftwf_plan& P) { fftwf_destroy_plan(P); }

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

void print_v(const char* name, MinAlignedVector v, size_t n) {
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

void test_fft_kernel_untimed(int64_t repeat_count, MinAlignedVector& Y_ref, MinAlignedVector& Y,
                             MinAlignedVector& X_ref, MinAlignedVector& X, MinAlignedVector& copy_X,
                             auto &P_ref, MinimalPlan* P, int64_t N, int32_t bm, double* t_ref_s,
                             double* t_s, int32_t num_factors, int64_t* Ns, fft_func_t* fns,
                             int32_t* es, void (*parent_fn)(void), int32_t inverse, double* std_dev,
                             const int64_t* params) {
  execute_fftw_plan(P_ref);
  if (P != nullptr) {
    P->execute_plan(Y, X, 0, 0, 1);
  } else {
    MFFTELEM* Y_data = Y.data();
    MFFTELEM* X_data = X.data();
    MFFTELEM** YY = &Y_data;
    MFFTELEM** XX = &X_data;
    if (parent_fn == nullptr) {
      ((fft_func_t)fns[0])(YY, XX, N, es[0], 0, 1, inverse);
    } else if (num_factors <= MAX_FACTORS) {
      ((parent_fn_t)parent_fn)(YY, XX, Ns, es, 0, 1, inverse, fns, params);
    } else {
      minassert(0, "Too many factors here.");
    }
    if (*YY != Y.data()) {
      std::swap(Y, X);
    }
    if (P!= nullptr) {
      if (approx_cmp_v(X_ref, X, N))
        minassert(0, "Planned FFT did not preserve input.");
    } else {
      X = copy_X;
    }
  }
  *std_dev = 0.0;
  *t_ref_s = 0.0;
  *t_s = 0.0;
}

void test_fft_kernel_timed(int64_t repeat_count, MinAlignedVector& Y_ref, MinAlignedVector& Y,
                           MinAlignedVector& X_ref, MinAlignedVector& X, MinAlignedVector& copy_X,
                           auto &P_ref, MinimalPlan* P, int64_t N, int32_t bm, double* t_ref_s,
                           double* t_s, int32_t num_factors, int64_t* Ns, fft_func_t* fns,
                           int32_t* es, void (*parent_fn)(void), int32_t inverse, double* std_dev,
                           const int64_t* params) {
  int64_t t_ref_start = 0, t_ref_end = 0;
  int64_t inner_start = 0, inner_end = 0;
  int64_t t_start = 0, t_end = 0;
  *t_ref_s = 0.0;
  *t_s = 0.0;
  MFFTELEM* Y_data = Y.data();
  MFFTELEM* X_data = X.data();
  MFFTELEM** YY = &Y_data;
  MFFTELEM** XX = &X_data;

  X = copy_X;
  X_ref = copy_X;
  t_ref_start = mingettime();
  execute_fftw_plan(P_ref);
  t_ref_end = mingettime();
  if (get_s_time(t_ref_start, t_ref_end) < 10e-6)
    repeat_count *= OVERSAMPLE_FACTOR;  // oversample if less than 10 us
  X_ref = copy_X;
  t_ref_start = mingettime();
  int64_t n = 0;
  while (n++ < repeat_count) {
    inner_start = mingettime();
    execute_fftw_plan(P_ref);
    X_ref = copy_X;
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
    if (P != nullptr) {
      P->execute_plan(Y, X, 0, 0, 1);
    } else {
      if (parent_fn == nullptr) {
        ((fft_func_t)fns[0])(YY, XX, N, es[0], 0, 1, inverse);
      } else if (num_factors <= MAX_FACTORS) {
        ((parent_fn_t)parent_fn)(YY, XX, Ns, es, 0, 1, inverse, fns, params);
      } else {
        minassert(0, "Too many factors here.");
      }
      if (*YY != Y.data()) {
        swap(Y, X);
      }
    }
    inner_end = mingettime();
    if (P!= nullptr) {
       if (approx_cmp_v(X_ref, X, N))
          minassert(0, "Planned FFT did not preserve input.");
    }
    else {
        X = copy_X; // restore input for next test
    }
    double x = get_s_time(inner_start, inner_end);
    delta = x - mu;
    mu += delta / (n + 1);
    delta2 = x - mu;
    M2 += delta * delta2;
  }
  t_end = mingettime();
  *std_dev = sqrt(1.0 * M2 / (n - 1));
  *t_ref_s = get_s_time(t_ref_start, t_ref_end) / (double)repeat_count;
  *t_s = get_s_time(t_start, t_end) / (double)repeat_count;
}

void test_fft_kernel(int64_t repeat_count, MinAlignedVector& Y_ref, MinAlignedVector& Y,
                     MinAlignedVector& X_ref, MinAlignedVector& X, MinAlignedVector& copy_X,
                     auto &P_ref, MinimalPlan* P, int64_t N, int32_t bm, double* t_ref_s,
                     double* t_s, int32_t num_factors, int64_t* Ns, fft_func_t* fns, int32_t* es,
                     void (*parent_fn)(void), int32_t inverse, double* std_dev,
                     const int64_t* params) {
  // dispatch time or untimed
  if (bm) {
    test_fft_kernel_timed(repeat_count, Y_ref, Y, X_ref, X, copy_X, P_ref, P, N, bm, t_ref_s, t_s,
                          num_factors, Ns, fns, es, parent_fn, inverse, std_dev, params);
  } else {
    test_fft_kernel_untimed(repeat_count, Y_ref, Y, X_ref, X, copy_X, P_ref, P, N, bm, t_ref_s, t_s,
                            num_factors, Ns, fns, es, parent_fn, inverse, std_dev, params);
  }
}

void print_result(const char* preamble, const char* name, int64_t N, int num_factors, int64_t* Ns,
                  int bm, double t_ref_s, double t_s, fft_func_t* fns, double std_dev) {
  char timing_str[256];
  timing_str[0] = '\0';
  if (bm) {
    snprintf(timing_str, sizeof(timing_str), "time = %2.2es xFFTW = %2.2e", t_s, t_s / t_ref_s);
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
    snprintf(factors_str + strlen(factors_str), sizeof(factors_str), "%" PRId64 "%c", Ns[i], comma);
  }
  if (bm) {
    printf("%s %s %s %s] std_dev=%2.2es (%s)\n", preamble, name, timing_str, factors_str, std_dev,
           fn_str);
  } else {
    printf("%s %s %s %s] (%s)\n", preamble, name, timing_str, factors_str, fn_str);
  }
  fflush(stdout);
}

void test_fft(random_normal& RNG, const char* name, int bm, int inverse, int64_t N, int* pc,
              int* fc, int num_factors, void (*parent_fn)(void), MinimalPlan* P, int64_t* Ns,
              fft_func_t* fns, int32_t* es) {
  struct timespec t_ref_start, t_ref_end;
  struct timespec t_start, t_end;
  double t_ref_s = DBL_MAX, t_s = DBL_MAX;
  MinAlignedVector Y_ref(N);
  MinAlignedVector Y(N);
  MinAlignedVector X_ref(RNG.get_rv(N));
  MinAlignedVector X(X_ref);
  MinAlignedVector copy_X(X_ref);
  auto P_ref = create_fftw_plan(N, X_ref.data(), Y_ref.data(), inverse);
  int test_repeat = bm ? NUM_TIMED_TESTS : 1;
  double std_dev;
  int64_t params[MAX_PFA_PARAMS] = {0};
  minassert(num_factors <= MAX_FACTORS, "Too many factors in test_fft.");
  if (num_factors > 1) generate_pfa_params(num_factors, Ns, params);
  test_fft_kernel(test_repeat, Y_ref, Y, X_ref, X, copy_X, P_ref, P, N, bm, &t_ref_s, &t_s,
                  num_factors, Ns, fns, es, parent_fn, inverse, &std_dev, params);
  // reporting
  if (bm) hm_add(&hm, t_s / t_ref_s);
  if (approx_cmp_v(Y_ref, Y, N)) {
    print_result("Failed for", name, N, num_factors, Ns, bm, t_ref_s, t_s, fns, std_dev);
    (*fc)++;
  } else {
    (*pc)++;
    print_result("Passed for", name, N, num_factors, Ns, bm, t_ref_s, t_s, fns, std_dev);
  }
  destroy_fftw_plan(P_ref);
}

void hex_dump(const void* ptr, size_t len) {
  const unsigned char* data = (const unsigned char*)ptr;
  for (size_t i = 0; i < len; ++i) {
    printf("%02X ", data[i]);
    if ((i + 1) % 16 == 0) printf("\n");
  }
  if (len % 16 != 0) printf("\n");
}

std::string gen_key(void (*parent_fn)(void), void* fns, int64_t* N_vals, int32_t* es, char bm,
                    int32_t inverse) {
  std::string key;
  key.append(reinterpret_cast<const char*>(&parent_fn), sizeof(void (*)()));
  key.append(reinterpret_cast<const char*>(fns), sizeof(void*) * MAX_FACTORS);
  key.append(reinterpret_cast<const char*>(N_vals), sizeof(int64_t) * MAX_FACTORS);
  key.append(reinterpret_cast<const char*>(es), sizeof(int32_t) * MAX_FACTORS);
  key.append(reinterpret_cast<const char*>(&bm), sizeof(char));
  key.append(reinterpret_cast<const char*>(&inverse), sizeof(inverse));
  return key;
}

int64_t prod(int64_t* arr, int len) {
  int64_t p = 1;
  for (int i = 0; i < len; ++i) {
    p *= arr[i];
  }
  return p;
}

void driver(random_normal& RNG, hashmap_t& d, int* radix, int radix_count, int num_factors, int bm,
            int* pc, int* fc, int32_t inverse, void (*parent_fn)(void), const int64_t* N_vals,
            int N_val_count, const char* name) {
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
            fns[i] = inverse ? dispatch_inverse[r] : dispatch[r];
          else if (factor < DIRECT_SZ)
            fns[i] = inverse ? &direct_dft<true> : &direct_dft<false>;
          else
            fns[i] = inverse ? &bluestein<true> : &bluestein<false>;
          i++;
          break;
        }
      }
      std::string key = gen_key((void (*)())(parent_fn), fns, Ns, es, bm, inverse);
      if (d.find(key) != d.end() || i != num_factors) continue;
      d[key] = 1;
      test_fft(RNG, name, bm, inverse, prod(Ns, num_factors), pc, fc, num_factors, parent_fn, 0, Ns,
               fns, es);
    }
  }
}

void print_time() {
  time_t now = time(NULL);
  struct tm* t = localtime(&now);
  char buf[64];
  buf[0] = '\0';
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
  printf("# Time Stamp: %s\n", buf);
}

void print_compiler_ver() {
  std::cout << "# CXX_COMPILER: " << BUILD_CXX_COMPILER << " ";
#ifdef __clang__
  std::cout << __clang_major__ << "." << __clang_minor__ << "."
            << __clang_patchlevel__;
#elif defined(__GNUC__)
  std::cout << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
  std::cout << _MSC_VER;
#else
  std::cout << "Unknown Compiler (" << __VERSION__ << ")";
#endif
  std::cout << " CXX_FLAGS: " << BUILD_CXX_FLAGS << std::endl;
  std::cout << "# BUILD_TYPE: " << BUILD_BUILD_TYPE << std::endl;
  std::cout << "# BUILD_SYSTEM: " << BUILD_SYSTEM << std::endl;
  int status = 0;
  char* demangled = abi::__cxa_demangle(typeid(MFFTELEM).name(), nullptr, nullptr, &status);
  std::cout << "# Vector type: " << demangled << std::endl << std::flush;
  free(demangled);
}

static const int64_t factor_1[][1] = {{8},  {4},  {25}, {27},        {16}, {125},
                                      {49}, {64}, {81}, {9 * 9 * 9}, {256}};

static const int64_t factor_2[][2] = {{4, 25}, {25, 4}, {4, 49}, {8, 9}, {256, 25}, {16, 5},
                                      {8, 7},  {11, 8}, {49, 3}, {9, 8}, {25, 256}, {1, 256}};

static const int64_t factor_3[][3] = {
    {9, 5, 49}, {9, 49, 5},    {5, 9, 49},    {49, 5, 9},    {8, 7, 25},   {7, 25, 8}, {2, 3, 5},
    {2, 5, 3},  {3, 2, 5},     {3, 5, 2},     {64, 3, 5},    {3, 5, 64},   {5, 64, 3}, {1, 1, 64},
    {64, 1, 1}, {27, 625, 49}, {625, 27, 49}, {49, 27, 625}, {49, 625, 27}};

static const int64_t high_radix_factor_1[][1] = {
    {262144}, {2097152},  // powers of 8
    {531441}, {4782969},  // powers of 9
    {78125},  {390625},   // powers of 5
    {823543}, {5764801}   // powers of 7
};

static const int64_t factor_4[][4] = {
    {3, 5, 7, 11}, {11, 7, 5, 3}, {25, 27, 49, 11}, {49, 27, 25, 11}, {49, 25, 27, 11},
    {7, 25, 9, 8}, {9, 25, 7, 8}, {8, 7, 25, 9},    {81, 49, 25, 11}, {1, 25, 49, 81}};

static const int64_t factor_5[][5] = {{3, 5, 7, 11, 13},  {13, 11, 7, 5, 3},   {25, 3, 49, 11, 13},
                                      {49, 9, 25, 11, 1}, {7, 25, 27, 11, 13}, {7, 25, 9, 2, 13},
                                      {9, 5, 7, 8, 13},   {8, 7, 5, 9, 13},    {17, 13, 81, 49, 5},
                                      {27, 5, 49, 11, 13}};

/*
static const int64_t factor_5[][5] = {{3, 5, 7, 11, 13},  {13, 11, 7, 5, 3},   {25, 3, 49, 11, 13},
                                      {49, 9, 25, 11, 1}, {7, 25, 27, 11, 13}, {7, 25, 9, 2, 13},
                                      {9, 5, 7, 8, 13},   {8, 7, 5, 9, 13},    
                                      81, 49, 5, 17, 13}, // breaks Bluestein sp
                                      {27, 5, 49, 11, 13}};
*/

static const int64_t factor_6[][6] = {{17, 4, 5, 7, 9, 11},
                                      {17, 11, 9, 7, 5, 4},
                                      {2, 3, 5, 7, 11, 13},
                                      {3, 2, 7, 5, 13, 11}};

static const int64_t factor_7[][7] = {{17, 4, 5, 7, 9, 11, 13},
                                      {17, 11, 9, 7, 5, 4, 13},
                                      {2, 3, 5, 7, 11, 13, 17},
                                      {3, 2, 7, 5, 13, 11, 17}};

/* original test cases                                     
static const int64_t factor_6[][6] = {{4, 5, 7, 9, 11, 17}, // breaks Bluestein sp
                                      {9, 7, 5, 4, 11, 17}, // breaks Bluestein sp 
                                      {2, 3, 5, 7, 11, 13},
                                      {3, 2, 7, 5, 13, 11}};
*/

static const int64_t bluestein_1[][1] = {{15}, {16}, {11}, {13}, {17}};

void bluestein_test_parent(MFFTELEM** YY, MFFTELEM** XX, const int64_t* Ns, const int32_t* es,
                           const int64_t bp, const int64_t stride, const int32_t flags,
                           const fft_func_t* fs, const int64_t* params) {
  if (flags & P_INVERSE) {
    bluestein<true>(YY, XX, Ns[0], es[0], bp, stride, flags);
  } else {
    bluestein<false>(YY, XX, Ns[0], es[0], bp, stride, flags);
  }
}

#ifdef __linux__

#include <fenv.h>
#include <arm_neon.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>

// Signal handler
void fpe_handler(int sig, siginfo_t *si, void *uc_void) {
    ucontext_t *uc = (ucontext_t *)uc_void;
    printf("Floating point exception trapped! si_code=%d\n", si->si_code);
    exit(EXIT_FAILURE);
}

void enable_fp_exceptions() {
    // Setup signal handler
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = fpe_handler;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGFPE, &sa, NULL) != 0) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    // Read FPCR
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));

    // Set Invalid Operation, Divide-by-zero, Overflow traps
    fpcr |= (1 << 8) | (1 << 9) | (1 << 10); // IOC | DZC | OFC

    // Write FPCR back
    asm volatile("msr fpcr, %0" :: "r"(fpcr));

    // Also enable exceptions in FE environment for C++ fenv
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW); // optional, may be stub
}

#endif

#ifndef VERSION
#define VERSION "unknown"
#endif

int main(int argc, char *argv[]) {
  hashmap_t d;
  std::random_device rd;
  random_normal RNG(rd(), 0.0, 1.0);

#ifdef __linux__
  enable_fp_exceptions();
#endif

  int pass = 0, fail = 0;
  int* pc = &pass;
  int* fc = &fail;

  int64_t t_test_start, t_test_end;
  t_test_start = mingettime();

  if (argc<2) {
    std::cerr << "test17: 1 or 0 for timing or no timing\n";
    exit(1);
  }

  int bm = -1;
  std::string bm_str(argv[1]);
  if (bm_str=="1")
    bm = 1;
  else if(bm_str=="0")
    bm = 0;
  if (bm==-1) {
    std::cerr << "test17: 1 or 0 for timing or no timing\n";
    exit(1);
  }

  hm_init(&hm);

  const char *version = VERSION;

  std::cout << "# test17 - MinimalFFT version: " << version << std::endl;
  print_time();
  print_compiler_ver();

    for (int n = 1; n <= SMALL_SZ; ++n) {
      if (small_available(n)) {
        int64_t N = n;
        fft_func_t fns[MAX_FACTORS] = {nullptr};
        fns[0] = &small_dft<false>;
        int32_t es = 1;
        test_fft(RNG, "small_dft", bm, 0, N, pc, fc, 1, NULL, nullptr, &N, fns, &es);
      }
    }

    for (int n = 1; n <= DIRECT_SZ; ++n) {
      int64_t N = n;
      fft_func_t fns[MAX_FACTORS] = {nullptr};
      fns[0] = &direct_dft<false>;
      int32_t es = 1;
      test_fft(RNG, "direct_dft", bm, 0, N, pc, fc, 1, NULL, nullptr, &N, fns, &es);
    }

  #define RUN_DRIVER(radix_arr, num_factors, bm, inverse, parent_fn, N_vals, name)                 \
    driver(RNG, d, (radix_arr), sizeof(radix_arr) / sizeof((radix_arr)[0]), (num_factors), bm, pc, \
           fc, (inverse), (void (*)())(parent_fn), &(N_vals)[0][0],                                \
           sizeof((N_vals)) / sizeof(N_vals[0]), (name))

    RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, bm, 0, nullptr, factor_1, "stockham test 0");
    RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, bm, 0, nullptr, factor_1, "stockham test 1");
    RUN_DRIVER(((int[]){2, 9, 5, 7}), 1, bm, 0, nullptr, factor_1, "stockham test 2");
    RUN_DRIVER(((int[]){4, 3, 5, 7}), 1, bm, 0, nullptr, factor_1, "stockham test 3");
    RUN_DRIVER(((int[]){4, 9, 5, 7}), 1, bm, 0, nullptr, factor_1, "stockham test 4");
    RUN_DRIVER(((int[]){8, 3, 5, 7}), 1, bm, 0, nullptr, factor_1, "stockham test 5");
    RUN_DRIVER(((int[]){8, 9, 5, 7}), 1, bm, 0, nullptr, factor_1, "stockham test 6");
    d.clear();
    RUN_DRIVER(((int[]){2}), 1, bm, 0, nullptr, factor_1, "timed stockham test 0");
    RUN_DRIVER(((int[]){3}), 1, bm, 0, nullptr, factor_1, "timed stockham test 1");
    RUN_DRIVER(((int[]){4}), 1, bm, 0, nullptr, factor_1, "timed stockham test 2");
    RUN_DRIVER(((int[]){5}), 1, bm, 0, nullptr, factor_1, "timed stockham test 3");
    RUN_DRIVER(((int[]){7}), 1, bm, 0, nullptr, factor_1, "timed stockham test 4");
    RUN_DRIVER(((int[]){8}), 1, bm, 0, nullptr, factor_1, "timed stockham test 5");
    RUN_DRIVER(((int[]){9}), 1, bm, 0, nullptr, factor_1, "timed stockham test 6");
    d.clear();
    RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, bm, 1, nullptr, factor_1, "stockham inverse test 0");
    RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, bm, 1, nullptr, factor_1, "stockham inverse test 1");
    RUN_DRIVER(((int[]){2, 9, 5, 7}), 1, bm, 1, nullptr, factor_1, "stockham inverse test 2");
    RUN_DRIVER(((int[]){4, 3, 5, 7}), 1, bm, 1, nullptr, factor_1, "stockham inverse test 3");
    RUN_DRIVER(((int[]){4, 9, 5, 7}), 1, bm, 1, nullptr, factor_1, "stockham inverse test 4");
    RUN_DRIVER(((int[]){8, 3, 5, 7}), 1, bm, 1, nullptr, factor_1, "stockham inverse test 5");
    RUN_DRIVER(((int[]){8, 9, 5, 7}), 1, bm, 1, nullptr, factor_1, "stockham inverse test 6");
    d.clear();
    RUN_DRIVER(((int[]){2, 3, 5, 7}), 2, 0, 1, prime_factor<2>, factor_2,
               "prime factor 2 test 0 inverse");
    RUN_DRIVER(((int[]){2, 3, 5, 7}), 2, bm, 0, prime_factor<2>, factor_2,
               "prime factor 2 test 1 timed");
    RUN_DRIVER(((int[]){4, 3, 5, 7}), 2, 0, 0, prime_factor<2>, factor_2, "prime factor 2 test 2");
    RUN_DRIVER(((int[]){8, 3, 5, 7}), 2, 0, 0, prime_factor<2>, factor_2, "prime factor 2 test 3");
    RUN_DRIVER(((int[]){2, 9, 5, 7}), 2, 0, 0, prime_factor<2>, factor_2, "prime factor 2 test 4");
    RUN_DRIVER(((int[]){8, 9, 5, 7}), 2, 0, 0, prime_factor<2>, factor_2, "prime factor 2 test 5");
    d.clear();
    RUN_DRIVER(((int[]){2, 3, 5, 7}), 3, 0, 1, prime_factor<3>, factor_3,
               "prime factor 3 test 0 inverse");
    RUN_DRIVER(((int[]){2, 3, 5, 7}), 3, bm, 0, prime_factor<3>, factor_3, "prime factor 3 test 1");
    RUN_DRIVER(((int[]){4, 3, 5, 7}), 3, 0, 0, prime_factor<3>, factor_3, "prime factor 3 test 2");
    RUN_DRIVER(((int[]){8, 3, 5, 7}), 3, 0, 0, prime_factor<3>, factor_3, "prime factor 3 test 3");
    RUN_DRIVER(((int[]){2, 9, 5, 7}), 3, 0, 0, prime_factor<3>, factor_3, "prime factor 3 test 4");
    RUN_DRIVER(((int[]){8, 9, 5, 7}), 3, 0, 0, prime_factor<3>, factor_3, "prime factor 3 test 5");
    d.clear();
    RUN_DRIVER(((int[]){2}), 1, 0, 0, nullptr, high_radix_factor_1, "radix 2 test");
    RUN_DRIVER(((int[]){3}), 1, 0, 0, nullptr, high_radix_factor_1, "radix 3 test");
    RUN_DRIVER(((int[]){4}), 1, 0, 0, nullptr, high_radix_factor_1, "radix 4 test");
    RUN_DRIVER(((int[]){5}), 1, 0, 0, nullptr, high_radix_factor_1, "radix 5 test");
    RUN_DRIVER(((int[]){7}), 1, 0, 0, nullptr, high_radix_factor_1, "radix 7 test");
    RUN_DRIVER(((int[]){8}), 1, 0, 0, nullptr, high_radix_factor_1, "radix 8 test");
    RUN_DRIVER(((int[]){9}), 1, 0, 0, nullptr, high_radix_factor_1, "radix 9 test");
    d.clear();
    RUN_DRIVER(((int[]){2, 3, 5, 7}), 4, 0, 0, prime_factor<4>, factor_4, "prime factor 4");
    RUN_DRIVER(((int[]){2, 3, 5, 7, 11, 13, 17}), 5, 0, 0, prime_factor<5>, factor_5,
               "prime factor 5");
    RUN_DRIVER(((int[]){2, 3, 5, 7, 11, 13, 17}), 6, 0, 0, prime_factor<6>, factor_6,
               "prime factor 6");
    RUN_DRIVER(((int[]){2, 3, 5, 7, 11, 13, 17}), 7, 0, 0, prime_factor<7>, factor_7,
               "prime factor 7");    
    d.clear();
    RUN_DRIVER(((int[]){15, 16, 11, 13, 17}), 1, 0, 0, bluestein_test_parent, bluestein_1,
               "bluestein test");
    RUN_DRIVER(((int[]){15, 16, 11, 13, 17}), 1, 0, 1, bluestein_test_parent, bluestein_1,
               "bluestein inverse test");
    d.clear();

  static int64_t planner_n[] = {4,
                                15,
                                30,
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
  int64_t* planner_n_inverse = planner_n;
  for (int i = 0; i < sizeof(planner_n) / sizeof(planner_n[0]); ++i) {
    int64_t N = planner_n[i];
    int64_t* N_p = &N;
    const int32_t region = 0;
    MinimalPlan P(N_p, 1, region, region, P_NONE);
    test_fft(RNG, "planner", bm, P_NONE, N, pc, fc, 1, nullptr, &P, &N, P.get_funcs(region),
             nullptr);
    MinimalPlan P_inv(N_p, 1, region, region, P_INVERSE);
    test_fft(RNG, "planner inverse", bm, P_INVERSE, N, pc, fc, 1, nullptr, &P_inv, &N,
             P_inv.get_funcs(region), nullptr);
  }
  printf("# Passed %d tests.\n", pass);
  printf("# Failed %d tests.\n", fail);
  if (bm) {
    t_test_end = mingettime();
    char timing_str[256];
    timing_str[0] = '\0';
    double elapsed = get_s_time(t_test_start,t_test_end);

    double hmv = hm_value(&hm);
    printf("# Total Time = %2.2e\n",elapsed);
    printf("# Mean xFFTW = %2.2e\n", m_value(&hm));
    printf("# Harmonic mean xFFTW = %2.2e\n", hmv);
    printf("# Geometric mean xFFTW = %2.2e\n", gm_value(&hm));
  }
  fflush(stdout);

  return 0;
}