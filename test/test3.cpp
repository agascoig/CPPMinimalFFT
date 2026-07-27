
// test3.cpp - do FFT repeatedly for profiling purposes

#include "common.hpp"
#include <vector>

extern struct MinimalPlanConfig MinPlanConfig;

void test_N(auto RNG, MinimalPlan& P, int64_t N, int64_t repeat_count) {
  MinAlignedVector X(RNG.get_rv(N));
  MinAlignedVector copy_X(X);
  MinAlignedVector Y(N);

  for (int64_t i = 0; i < repeat_count; ++i) {
    P.execute_plan(Y, X, 0, 0, 1);
  }

}

int main(int argc, char **argv) {
  
  const char* version = VERSION;

  std::cout << "# test3 - MinimalFFT version: " << version << std::endl;
  print_time();
  print_compiler_ver();

  std::random_device rd;
  random_normal RNG(rd(), 0.0, 1.0);

  int64_t N = 4 * 27 * 5 * 7;
  int64_t repeat_count = 1000000;
  int64_t t_start = 0, t_end = 0;

  MinimalPlan P(&N, 1, 0, 0, P_NONE);
  std::cout << "P: " << P << std::endl;
  t_start=mingettime();
  test_N(RNG, P, N, repeat_count);
  t_end=mingettime();

  double elapsed_s = get_s_time(t_start, t_end);
  std::cout << "test3: N=" << N << " repeat_count=" << repeat_count << " elapsed_s=" << elapsed_s;

  double mbsec = sizeof(MFFTELEM) * N * repeat_count / (1e6 * elapsed_s);

  std::cout << " MB/s " << mbsec << std::endl;

  fflush(stdout);
}