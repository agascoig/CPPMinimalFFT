
// test4.cpp - do FFT repeatedly comparing to FFTW

#include "common.hpp"
#include <vector>

int main(int argc, char **argv) {
  
  const char* version = VERSION;

  std::cout << "# test4 - MinimalFFT version: " << version << std::endl;
  print_time();
  print_compiler_ver();

  std::random_device rd;
  random_normal RNG(rd(), 0.0, 1.0);

  int64_t N = 3780;
  int64_t repeat_count = 2000;
  int64_t t_start = 0, t_end = 0;

  MinimalPlan P(&N, 1, 0, 0, P_NONE);
  std::cout << "P: " << P << std::endl;

  int pass = 0, fail = 0;

  for (int i = 0; i < repeat_count; ++i) { 
    test_fft(RNG, "test4", 1, 0, N, &pass, &fail, 0, 1, false, &P, &N, nullptr, nullptr,
             nullptr);
  }

}