
// test1.cpp - simple butterfly test (multiply with identity for DFT matrix)

#include "common.hpp"
#include <vector>

bool test_column(bool inverse, MinimalPlan& P, int64_t N, int column) {
  MinAlignedVector X(N);
  MinAlignedVector Y(N);
  X[column] = 1;

  MinAlignedVector Y_ref(N);
  for (int i = 0; i < N; ++i) {
    if (inverse)
      Y_ref[i] = minsincos(2 * M_PI * column * i / N);
    else
      Y_ref[i] = minsincos(-2 * M_PI * i * column / N);
  }

  P.execute_plan(Y, X, 0, 0, 1);

//  print_v("Y_ref",Y_ref,N);
//  print_v("Y", Y, N);

  if (approx_cmp_v(Y_ref, Y, N)) {
    for (int k=0;k<N;++k) {
      if (approx_cmp(Y_ref[k],Y[k])) {
         std::cout << "(" << column+1 << "," << k+1 << "): " << Y_ref[k] << " " << Y[k] << std::endl;
      }
    }
    return false;
  } else {
    return true;
  }
}

int main(int argc, char* argv[]) {
  int pass = 0, fail = 0;
  int* pc = &pass;
  int* fc = &fail;

  const char* version = VERSION;

  std::cout << "# test1 - MinimalFFT version: " << version << std::endl;
  print_time();
  print_compiler_ver();

  // butterfly sizes
  static int64_t planner_n[] = {2, 3, 4, 5, 7, 8, 9, 11, 13, 17, 19, 23, 29, 31};
  auto planner_n_inverse = planner_n;

  int num_tests = sizeof(planner_n) / sizeof(planner_n[0]);
  for (int i = 0; i < num_tests; ++i) {
    int64_t N = planner_n[i];
    MinimalPlan P(&N, 1, 0, 0, P_NONE, 0, 0);
    std::cout << "--------\nP=" << P << std::endl;
    bool success = true;
    for (int column = 0; column < N; ++column) {
      bool result = test_column(false, P, N, column);
      if (!result) {
        print_result("Test failed", "butterfly", N, 1, &N, 0, 0.0, 0.0, nullptr, 0.0);
        std::cout << "   First column failed: " << column+1 << std::endl;
        std::cout << std::endl;
        success = false;
        (*fc)++;
        break;
      }
    }
    if (success) {
      print_result("Test passed", "butterfly", N, 1, &N, 0, 0.0, 0.0, nullptr, 0.0);
      (*pc)++;
    }

    success = true;
    MinimalPlan P_inv(&N, 1, 0, 0, P_INVERSE, 0, 0);
    std::cout << "--------\nP_inv=" << P_inv << std::endl;
    for (int column = 0; column < N; ++column) {
      bool result = test_column(true, P_inv, N, column);
      if (!result) {
        print_result("Test failed ", "butterfly inverse", N, 1, &N, 0, 0.0, 0.0, nullptr, 0.0);
        std::cout << "   First column failed: " << column+1 << std::endl;
        success = false;
        (*fc)++;
        break;
      }
    }
    if (success) {
      print_result("Test passed", "butterfly inverse", N, 1, &N, 0, 0.0, 0.0, nullptr, 0.0);
      (*pc)++;
    }
  }

  printf("# Passed %d tests.\n", pass);
  printf("# Failed %d tests.\n", fail);
  fflush(stdout);

  return 0;
}