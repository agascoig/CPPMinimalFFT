
// test1.cpp - sweep test N=1..(37)^3

#include "common.hpp"
#include <vector>

extern struct MinimalPlanConfig MinPlanConfig;

bool test_N(auto RNG, MinimalPlan& P, int64_t N, bool inverse) {
  MinAlignedVector X(N);
  MinAlignedVector Y(N);
  MinAlignedVector X_ref(RNG.get_rv(N));
  X = X_ref;
  MinAlignedVector Y_ref(N);

  auto P_ref = create_fftw_plan(N, X_ref.data(), Y_ref.data(), inverse);
  execute_fftw_plan(P_ref);
  destroy_fftw_plan(P_ref);

  P.execute_plan(Y, X, 0, 0, 1);

  if (approx_cmp_v(Y_ref, Y, N)) {
    print_v("Y_ref", Y_ref, N);
    print_v("Y", Y, N);
    return false;
  } else {
    return true;
  }
}

int driver(auto RNG, int &pass, int &fail, int N_limit, int _direct_sz, int _small_sz) {
  // planner control
  MinPlanConfig.direct_sz = _direct_sz;
  MinPlanConfig.small_sz = _small_sz;

  for (int64_t N = 1; N <= N_limit; ++N) {
    MinimalPlan P(&N, 1, 0, 0, P_NONE);
    bool result = test_N(RNG, P, N, false);
    if (!result) {
      std::cout << "Forward plan failed for N=" << N << " " << P << std::endl;
      print_result("Test failed", "forward plan", N, 1, &N, 0, 0.0, 0.0, nullptr, 0.0);
      fail++;
      break; // optional
    }
    else {
//      print_result("Test passed", "forward plan", N, 1, &N, 0, 0.0, 0.0, nullptr, 0.0);
      pass++;
    }
    MinimalPlan P_inv(&N, 1, 0, 0, P_INVERSE);
    bool result_inv = test_N(RNG, P_inv, N, true);
    if (!result_inv) {
        std::cout << "Inverase plan failed for N=" << N << " " << P << std::endl;
      print_result("Test failed", "inverse plan", N, 1, &N, 0, 0.0, 0.0, nullptr, 0.0);
      fail++;
      break; // optional
    } else {
//      print_result("Test passed", "inverse plan", N, 1, &N, 0, 0.0, 0.0, nullptr, 0.0);
      pass++;
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  int pass = 0, fail = 0;
  
  const char* version = VERSION;

  std::cout << "# test2 - MinimalFFT version: " << version << std::endl;
  print_time();
  print_compiler_ver();

  std::random_device rd;
  random_normal RNG(rd(), 0.0, 1.0);

  int max = DEFAULT_SMALL_SZ > DEFAULT_DIRECT_SZ ? DEFAULT_SMALL_SZ : DEFAULT_DIRECT_SZ;

  std::cout << "Sweeping forward and inverse direct, small off N=1.." << max << std::endl;
  driver(RNG, pass, fail, max, 0, 0);
  std::cout << "Sweeping forward and inverse direct N=1..." << max << std::endl;
  driver(RNG, pass, fail, max, DEFAULT_DIRECT_SZ, 0);
  std::cout << "Sweeping forward and inverse small N=1..." << max << std::endl;
  driver(RNG, pass, fail, max, 0, DEFAULT_SMALL_SZ);
  std::cout << "Sweeping forward and inverse N=1..50653\n";
  driver(RNG, pass, fail, 37*37*37, DEFAULT_DIRECT_SZ, DEFAULT_SMALL_SZ);

  printf("# Passed %d tests.\n", pass);
  printf("# Failed %d tests.\n", fail);
  fflush(stdout);
}