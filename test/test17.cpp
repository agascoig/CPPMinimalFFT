
#include "common.hpp"

static const int64_t DIRECT_SZ = 15;

static const int64_t factor_1[][1] = {{8},       {4},       {25},           {27},      {16},
                                      {125},     {49},      {64},           {81},      {9 * 9 * 9},
                                      {256},     {11 * 11}, {11 * 11 * 11}, {13 * 13}, {17 * 17},
                                      {19 * 19}, {23 * 23}, {29 * 29},      {31 * 31}};

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

// note: large ending factors break bluestein single-precision SIMD
static const int64_t factor_5[][5] = {{3, 5, 7, 11, 13},  {13, 11, 7, 5, 3},   {25, 3, 49, 11, 13},
                                      {49, 9, 25, 11, 1}, {7, 25, 27, 11, 13}, {7, 25, 9, 2, 13},
                                      {9, 5, 7, 8, 13},   {8, 7, 5, 9, 13},    {17, 13, 81, 49, 5},
                                      {27, 5, 49, 11, 13}};

// note: large ending factors break bluestein single-precision SIMD
static const int64_t factor_6[][6] = {
    {17, 4, 5, 7, 9, 11}, {17, 11, 9, 7, 5, 4}, {2, 3, 5, 7, 11, 13}, {3, 2, 7, 5, 13, 11}};

static const int64_t factor_7[][7] = {{17, 4, 5, 7, 9, 11, 13},
                                      {17, 11, 9, 7, 5, 4, 13},
                                      {17, 13, 2, 3, 5, 7, 11},
                                      {17, 11, 3, 2, 7, 5, 13}};

static const int64_t bluestein_1[][1] = {{37}, {41}, {43}, {47}};

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

int main(int argc, char* argv[]) {
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

  if (argc < 2) {
    std::cerr << "test17: 1 or 0 for timing or no timing\n";
    exit(1);
  }

  int bm = -1;
  std::string bm_str(argv[1]);
  if (bm_str == "1")
    bm = 1;
  else if (bm_str == "0")
    bm = 0;
  if (bm == -1) {
    std::cerr << "test17: 1 or 0 for timing or no timing\n";
    exit(1);
  }

  hm_init(&hm);

  const char* version = VERSION;

  std::cout << "# test17 - MinimalFFT version: " << version << std::endl;
  print_time();
  print_compiler_ver();

  for (int n = 1; n <= SMALL_SZ; ++n) {
    if (small_available(n)) {
      int64_t N = n;
      fft_func_t fns[MAX_FACTORS] = {nullptr};
      fns[0] = &small_dft<false>;
      int32_t es = 1;
      test_fft(RNG, "small_dft", bm, 0, N, pc, fc, 1, 1, false, nullptr, &N, fns, &es);
    }
  }

  for (int n = 1; n <= DIRECT_SZ; ++n) {
    int64_t N = n;
    fft_func_t fns[MAX_FACTORS] = {nullptr};
    fns[0] = &direct_dft<false>;
    int32_t es = 1;
    test_fft(RNG, "direct_dft", bm, 0, N, pc, fc, 1, 1, false, nullptr, &N, fns, &es);
  }

#define RUN_DRIVER(radix_arr, nf, bm, inverse, pfa, N_vals, name)                           \
  driver(RNG, d, (radix_arr), sizeof(radix_arr) / sizeof((radix_arr)[0]), (nf), bm, pc, fc, \
         (inverse), pfa, &(N_vals)[0][0], sizeof((N_vals)) / sizeof(N_vals[0]), (name), DIRECT_SZ)

  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, bm, 0, false, factor_1, "stockham test 0");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, bm, 0, false, factor_1, "stockham test 1");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 1, bm, 0, false, factor_1, "stockham test 2");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 1, bm, 0, false, factor_1, "stockham test 3");
  RUN_DRIVER(((int[]){4, 9, 5, 7}), 1, bm, 0, false, factor_1, "stockham test 4");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 1, bm, 0, false, factor_1, "stockham test 5");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 1, bm, 0, false, factor_1, "stockham test 6");
  d.clear();
  RUN_DRIVER(((int[]){2}), 1, bm, 0, false, factor_1, "timed stockham test 0");
  RUN_DRIVER(((int[]){3}), 1, bm, 0, false, factor_1, "timed stockham test 1");
  RUN_DRIVER(((int[]){4}), 1, bm, 0, false, factor_1, "timed stockham test 2");
  RUN_DRIVER(((int[]){5}), 1, bm, 0, false, factor_1, "timed stockham test 3");
  RUN_DRIVER(((int[]){7}), 1, bm, 0, false, factor_1, "timed stockham test 4");
  RUN_DRIVER(((int[]){8}), 1, bm, 0, false, factor_1, "timed stockham test 5");
  RUN_DRIVER(((int[]){9}), 1, bm, 0, false, factor_1, "timed stockham test 6");
  RUN_DRIVER(((int[]){11}), 1, bm, 0, false, factor_1, "timed stockham test 7");
  RUN_DRIVER(((int[]){13}), 1, bm, 0, false, factor_1, "timed stockham test 8");
  RUN_DRIVER(((int[]){17}), 1, bm, 0, false, factor_1, "timed stockham test 9");
  RUN_DRIVER(((int[]){19}), 1, bm, 0, false, factor_1, "timed stockham test 10");
  RUN_DRIVER(((int[]){23}), 1, bm, 0, false, factor_1, "timed stockham test 11");
  RUN_DRIVER(((int[]){29}), 1, bm, 0, false, factor_1, "timed stockham test 12");
  RUN_DRIVER(((int[]){31}), 1, bm, 0, false, factor_1, "timed stockham test 13");

  d.clear();
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, bm, 1, false, factor_1, "stockham inverse test 0");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, bm, 1, false, factor_1, "stockham inverse test 1");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 1, bm, 1, false, factor_1, "stockham inverse test 2");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 1, bm, 1, false, factor_1, "stockham inverse test 3");
  RUN_DRIVER(((int[]){4, 9, 5, 7}), 1, bm, 1, false, factor_1, "stockham inverse test 4");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 1, bm, 1, false, factor_1, "stockham inverse test 5");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 1, bm, 1, false, factor_1, "stockham inverse test 6");
  d.clear();
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 2, 0, 1, true, factor_2, "prime factor 2 test 0 inverse");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 2, bm, 0, true, factor_2, "prime factor 2 test 1 timed");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 2, 0, 0, true, factor_2, "prime factor 2 test 2");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 2, 0, 0, true, factor_2, "prime factor 2 test 3");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 2, 0, 0, true, factor_2, "prime factor 2 test 4");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 2, 0, 0, true, factor_2, "prime factor 2 test 5");
  d.clear();
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 3, 0, 1, true, factor_3, "prime factor 3 test 0 inverse");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 3, bm, 0, true, factor_3, "prime factor 3 test 1");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 3, 0, 0, true, factor_3, "prime factor 3 test 2");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 3, 0, 0, true, factor_3, "prime factor 3 test 3");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 3, 0, 0, true, factor_3, "prime factor 3 test 4");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 3, 0, 0, true, factor_3, "prime factor 3 test 5");
  d.clear();
  RUN_DRIVER(((int[]){2}), 1, 0, 0, false, high_radix_factor_1, "radix 2 test");
  RUN_DRIVER(((int[]){3}), 1, 0, 0, false, high_radix_factor_1, "radix 3 test");
  RUN_DRIVER(((int[]){4}), 1, 0, 0, false, high_radix_factor_1, "radix 4 test");
  RUN_DRIVER(((int[]){5}), 1, 0, 0, false, high_radix_factor_1, "radix 5 test");
  RUN_DRIVER(((int[]){7}), 1, 0, 0, false, high_radix_factor_1, "radix 7 test");
  RUN_DRIVER(((int[]){8}), 1, 0, 0, false, high_radix_factor_1, "radix 8 test");
  RUN_DRIVER(((int[]){9}), 1, 0, 0, false, high_radix_factor_1, "radix 9 test");
  d.clear();
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 4, 0, 0, true, factor_4, "prime factor 4");
  RUN_DRIVER(((int[]){2, 3, 5, 7, 11, 13, 17}), 5, 0, 0, true, factor_5, "prime factor 5");
  RUN_DRIVER(((int[]){2, 3, 5, 7, 11, 13, 17}), 6, 0, 0, true, factor_6, "prime factor 6");
  RUN_DRIVER(((int[]){2, 3, 5, 7, 11, 13, 17}), 7, 0, 0, true, factor_7, "prime factor 7");
  d.clear();
  RUN_DRIVER(((int[]){15, 16, 11, 13, 17}), 1, 0, 0, false, bluestein_1, "bluestein test");
  RUN_DRIVER(((int[]){15, 16, 11, 13, 17}), 1, 0, 1, false, bluestein_1, "bluestein inverse test");
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
                                9 * 2 * 7 * 5 * 11 * 13 * 17,  // bluestein reorder test
                                1 << 20,
                                1 << 22};
  int64_t* planner_n_inverse = planner_n;
  for (int i = 0; i < sizeof(planner_n) / sizeof(planner_n[0]); ++i) {
    int64_t N = planner_n[i];
    int64_t* N_p = &N;
    const int32_t region = 0;
    MinimalPlan P(N_p, 1, region, region, P_NONE);
    test_fft(RNG, "planner", bm, P_NONE, N, pc, fc, 1, 1, false, &P, &N, P.get_funcs(region),
             nullptr);
    MinimalPlan P_inv(N_p, 1, region, region, P_INVERSE);
    test_fft(RNG, "planner inverse", bm, P_INVERSE, N, pc, fc, 1, 1, false, &P_inv, &N,
             P_inv.get_funcs(region), nullptr);
  }

  printf("# Passed %d tests.\n", pass);
  printf("# Failed %d tests.\n", fail);
  if (bm) {
    t_test_end = mingettime();
    char timing_str[256];
    timing_str[0] = '\0';
    double elapsed = get_s_time(t_test_start, t_test_end);

    double hmv = hm_value(&hm);
    printf("# Total Time = %2.2e\n", elapsed);
    printf("# Mean xFFTW = %2.2e\n", m_value(&hm));
    printf("# Harmonic mean xFFTW = %2.2e\n", hmv);
    printf("# Geometric mean xFFTW = %2.2e\n", gm_value(&hm));
  }
  fflush(stdout);

  return 0;
}