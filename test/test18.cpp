
// test18.cpp - multi-dimensional planned FFT tests

#include "common.hpp"

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
    std::cerr << "test18: 1 or 0 for timing or no timing\n";
    exit(1);
  }

  int bm = -1;
  std::string bm_str(argv[1]);
  if (bm_str == "1")
    bm = 1;
  else if (bm_str == "0")
    bm = 0;
  if (bm == -1) {
    std::cerr << "test18: 1 or 0 for timing or no timing\n";
    exit(1);
  }

  hm_init(&hm);

  const char* version = VERSION;

  std::cout << "# test18 - MinimalFFT version: " << version << std::endl;
  print_time();
  print_compiler_ver();

  // multi-dimensional tests

  static const int MAX_TEST_DIMS = 5;
  static int64_t planner_multid[][MAX_TEST_DIMS] = {{2,2,0,0,0}, 
  {16,3,0,0,0}, {8,9,0,0,0}, {2,3,5,0,0}, {2, 3, 5, 7, 0},
  {4, 3, 5, 7, 0}, {27, 8, 5, 16, 0}, {2,3,5,7,9}, {4,3,49}};
  auto planner_multid_inverse = planner_multid;

  int num_tests = sizeof(planner_multid) / sizeof(planner_multid[0]);
  for (int i = 0; i < num_tests; ++i) {
    int64_t *Ns = &planner_multid[i][0];
    int dims = 0;
    while (dims < MAX_TEST_DIMS && Ns[dims] != 0)
       dims++;
    auto N = prod(dims, Ns);
    const int32_t region_start = 0;
    const int32_t region_end = dims-1;
    MinimalPlan P(Ns, dims, region_start, region_end, P_NONE);
    test_fft(RNG, "planner multid", bm, P_NONE, N, pc, fc, 1, dims, false, &P, Ns, nullptr,
             nullptr);
    MinimalPlan P_inv(Ns, dims, region_start, region_end, P_INVERSE);
    test_fft(RNG, "planner multid inverse", bm, P_INVERSE, N, pc, fc, 1, dims, false, &P_inv, Ns,
             nullptr, nullptr);
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