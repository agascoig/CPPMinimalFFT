
#pragma once

#include <buildinfo.h>
#include <cxxabi.h>
#include <fftw3.h>
#include <stdio.h>

#include <cfenv>  // for NaN trapping
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

#include "CPPMinimalFFT.hpp"
#include "hmean.hpp"
#include "pfa.hpp"
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

// here due to MAX_FACTORS
static void print_fns(char* buf, fft_func_t* fns) {
  char fn_str[256];
  buf[0] = '\0';
  fn_str[0] = '\0';
  if (fns == nullptr) return;
  for (int i = 0; i < MAX_FACTORS; ++i) {
    fft_func_t func = fns[i];
    if (func == nullptr) continue;
    for (int j = 0; j < sizeof(fns_names) / sizeof(fn_name_s); ++j) {
      if (func == fns_names[j].fn) {
        snprintf(fn_str, sizeof(fn_str), "%s ", fns_names[j].name);
        strcat(buf, fn_str);
        break;
      }
    }
  }
  if (strlen(buf)) buf[strlen(buf) - 1] = 0;  // remove last space
}

fftwf_plan create_fftw_plan(int n, std::complex<float>* in, std::complex<float>* out, int inverse) {
  auto dir = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
  auto P = fftwf_plan_dft_1d(n, (fftwf_complex*)in, (fftwf_complex*)out, dir, FFTW_ESTIMATE);
  minassert(P != nullptr, "Failed to generate FFTW plan.");
  return P;
}

fftw_plan create_fftw_plan(int n, std::complex<double>* in, std::complex<double>* out,
                           int inverse) {
  auto dir = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
  auto P = fftw_plan_dft_1d(n, (fftw_complex*)in, (fftw_complex*)out, dir, FFTW_ESTIMATE);
  minassert(P != nullptr, "Failed to generate FFTW plan.");
  return P;
}

fftwf_plan create_fftw_multid_plan(int dims, int64_t* Ns, std::complex<float>* in,
                                   std::complex<float>* out, int inverse) {
  int* Ns_rev = new int[dims];  // reverse for row-major FFTW
  for (int i = 0; i < dims; ++i) Ns_rev[i] = Ns[dims - i - 1];
  auto dir = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
  auto P =
      fftwf_plan_dft(dims, Ns_rev, (fftwf_complex*)in, (fftwf_complex*)out, dir, FFTW_ESTIMATE);
  minassert(P != nullptr, "Failed to generate FFTW plan.");
  delete[] Ns_rev;  // safe
  return P;
}

fftw_plan create_fftw_multid_plan(int dims, int64_t* Ns, std::complex<double>* in,
                                  std::complex<double>* out, int inverse) {
  int* Ns_rev = new int[dims];  // reverse for row-major FFTW
  for (int i = 0; i < dims; ++i) Ns_rev[i] = Ns[dims - i - 1];
  auto dir = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
  auto P = fftw_plan_dft(dims, Ns_rev, (fftw_complex*)in, (fftw_complex*)out, dir, FFTW_ESTIMATE);
  minassert(P != nullptr, "Failed to generate FFTW plan.");
  delete[] Ns_rev;  // safe
  return P;
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
                             auto& P_ref, MinimalPlan* P, int64_t N, int32_t bm, double* t_ref_s,
                             double* t_s, int32_t nf, int64_t* Ns, fft_func_t* fns, int32_t* es,
                             bool pfa, int32_t inverse, double* std_dev, const int64_t* QPs,
                             const MAP_CACHE_T* nm, const MAP_CACHE_T* km) {
  execute_fftw_plan(P_ref);
  if (P != nullptr) {
    int r_start = P->get_region_start();
    int r_end = P->get_region_end();
    if (r_end==0)
      P->execute_plan(Y,X, 0, 0, 1);
      else
      P->execute_multid_plan(Y, X, r_start, r_end, 0, 1);
  } else {
    MFFTELEM* Y_data = Y.data();
    MFFTELEM* X_data = X.data();
    MFFTELEM** YY = &Y_data;
    MFFTELEM** XX = &X_data;
    if (!pfa) {
      ((fft_func_t)fns[0])(YY, XX, N, es[0], 0, 1, inverse);
    } else if (nf <= MAX_FACTORS) {
      prime_factor(nf, YY, XX, N, Ns, es, 0, 1, inverse, fns, QPs, nm, km);
    } else {
      minassert(0, "Too many factors here.");
    }
    if (*YY != Y.data()) {
      std::swap(Y, X);
    }
    if (P != nullptr) {
      if (approx_cmp_v(X_ref, X, N)) minassert(0, "Planned FFT did not preserve input.");
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
                           auto& P_ref, MinimalPlan* P, int64_t N, int32_t bm, double* t_ref_s,
                           double* t_s, int32_t nf, int64_t* Ns, fft_func_t* fns, int32_t* es,
                           bool pfa, int32_t inverse, double* std_dev, const int64_t* QPs,
                           const MAP_CACHE_T* nm, const MAP_CACHE_T* km) {
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
      int r_start = P->get_region_start();
      int r_end = P->get_region_end();
      if (r_end == 0)
        P->execute_plan(Y, X, 0, 0, 1);
      else {
        P->execute_multid_plan(Y, X, r_start, r_end, 0, 1);
      }
    } else {
      if (!pfa) {
        ((fft_func_t)fns[0])(YY, XX, N, es[0], 0, 1, inverse);
      } else if (nf <= MAX_FACTORS) {
        prime_factor(nf, YY, XX, N, Ns, es, 0, 1, inverse, fns, QPs, nm, km);
      } else {
        minassert(0, "Too many factors here.");
      }
      if (*YY != Y.data()) {
        swap(Y, X);
      }
    }
    inner_end = mingettime();
    if (P != nullptr) {
      if (approx_cmp_v(X_ref, X, N)) minassert(0, "Planned FFT did not preserve input.");
    } else {
      X = copy_X;  // restore input for next test
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
                     auto& P_ref, MinimalPlan* P, int64_t N, int32_t bm, double* t_ref_s,
                     double* t_s, int32_t nf, int64_t* Ns, fft_func_t* fns, int32_t* es, bool pfa,
                     int32_t inverse, double* std_dev, const int64_t* QPs, const MAP_CACHE_T* nm,
                     const MAP_CACHE_T* km) {
  // dispatch time or untimed
  if (bm) {
    test_fft_kernel_timed(repeat_count, Y_ref, Y, X_ref, X, copy_X, P_ref, P, N, bm, t_ref_s, t_s,
                          nf, Ns, fns, es, pfa, inverse, std_dev, QPs, nm, km);
  } else {
    test_fft_kernel_untimed(repeat_count, Y_ref, Y, X_ref, X, copy_X, P_ref, P, N, bm, t_ref_s, t_s,
                            nf, Ns, fns, es, pfa, inverse, std_dev, QPs, nm, km);
  }
}

void print_result(const char* preamble, const char* name, int64_t N, int nf, int64_t* Ns, int bm,
                  double t_ref_s, double t_s, fft_func_t* fns, double std_dev) {
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
  for (int i = 0; i < nf; i++) {
    if (i == (nf - 1)) comma = '\0';
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
              int* fc, int nf, int dims, bool pfa, MinimalPlan* P, int64_t* Ns, fft_func_t* fns,
              int32_t* es) {
  struct timespec t_ref_start, t_ref_end;
  struct timespec t_start, t_end;
  double t_ref_s = DBL_MAX, t_s = DBL_MAX;
  MinAlignedVector Y_ref(N);
  MinAlignedVector Y(N);
  MinAlignedVector X_ref(RNG.get_rv(N));
  MinAlignedVector X(X_ref);
  MinAlignedVector copy_X(X_ref);
  fftw_plan P_ref;
  if (dims > 1) {
    P_ref = create_fftw_multid_plan(dims, Ns, X_ref.data(), Y_ref.data(), inverse);
  } else
    P_ref = create_fftw_plan(N, X_ref.data(), Y_ref.data(), inverse);
  int test_repeat = bm ? NUM_TIMED_TESTS : 1;
  double std_dev;
  minassert(nf <= MAX_FACTORS, "Too many factors in test_fft.");
  if (nf > 1) {
    int64_t* QPs = P ? nullptr : generate_QPs(nf, Ns);
    MAP_CACHE_T* nm = P ? nullptr : generate_nmap(nf, N, Ns, QPs);
    MAP_CACHE_T* km = P ? nullptr : generate_kmap(nf, N, Ns, QPs);
    test_fft_kernel(test_repeat, Y_ref, Y, X_ref, X, copy_X, P_ref, P, N, bm, &t_ref_s, &t_s, nf,
                    Ns, fns, es, true, inverse, &std_dev, QPs, nm, km);
    if (QPs) delete[] QPs;
    if (nm) delete[] nm;
    if (km) delete[] km;
  } else {
    test_fft_kernel(test_repeat, Y_ref, Y, X_ref, X, copy_X, P_ref, P, N, bm, &t_ref_s, &t_s, nf,
                    Ns, fns, es, false, inverse, &std_dev, nullptr, nullptr, nullptr);
  }
  // reporting
  if (bm) hm_add(&hm, t_s / t_ref_s);
  if (approx_cmp_v(Y_ref, Y, N)) {
    print_result("Failed for", name, N, nf, Ns, bm, t_ref_s, t_s, fns, std_dev);
    (*fc)++;
  } else {
    (*pc)++;
    print_result("Passed for", name, N, nf, Ns, bm, t_ref_s, t_s, fns, std_dev);
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

std::string gen_key(bool pfa, void* fns, int64_t* N_vals, int32_t* es, char bm, int32_t inverse) {
  std::string key;
  key.append(reinterpret_cast<const char*>(&pfa), sizeof(bool));
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

void driver(random_normal& RNG, hashmap_t& d, int* radix, int radix_count, int nf, int bm, int* pc,
            int* fc, int32_t inverse, bool pfa, const int64_t* N_vals, int N_val_count,
            const char* name) {
  int64_t bs[MAX_FACTORS];
  int32_t es[MAX_FACTORS];
  fft_func_t fns[MAX_FACTORS];
  int64_t Ns[MAX_FACTORS];
  for (int t = 0; t < N_val_count; t++) {
    int64_t i = 0;
    memset(fns, 0, MAX_FACTORS * sizeof(fft_func_t));
    memset(Ns, 0, MAX_FACTORS * sizeof(int64_t));
    memset(es, 0, MAX_FACTORS * sizeof(int32_t));
    minassert(nf <= MAX_FACTORS, "Too many factors.");
    for (int f = 0; f < nf; f++) {
      const int64_t factor = N_vals[t * nf + f];
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
      std::string key = gen_key(pfa, fns, Ns, es, bm, inverse);
      if (d.find(key) != d.end() || i != nf) continue;
      d[key] = 1;
      test_fft(RNG, name, bm, inverse, prod(Ns, nf), pc, fc, nf, 1, pfa, 0, Ns, fns, es);
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
  std::cout << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
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

#ifdef __linux__

#include <arm_neon.h>
#include <fenv.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>

// Signal handler
void fpe_handler(int sig, siginfo_t* si, void* uc_void) {
  ucontext_t* uc = (ucontext_t*)uc_void;
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
  fpcr |= (1 << 8) | (1 << 9) | (1 << 10);  // IOC | DZC | OFC

  // Write FPCR back
  asm volatile("msr fpcr, %0" ::"r"(fpcr));

  // Also enable exceptions in FE environment for C++ fenv
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);  // optional, may be stub
}

#endif

#ifndef VERSION
#define VERSION "unknown"
#endif
