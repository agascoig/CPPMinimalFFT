
#ifndef __PLAN_H__
#define __PLAN_H__

#include "CPPMinimalFFT.hpp"
#include <stdint.h>

static const int P_NONE = 0;
static const int P_INVERSE = 1;
static const int P_INPLACE = 2;
static const int P_REAL = 4;
static const int P_ISBFFT = 8;
static const int P_ODD = 16;
static const int P_SCALED = 32;
static const int P_TOO_MANY_FACTORS = 64;

static const int DIRECT_SZ = 15;
static const int MAX_FACTORS = 6;

// Prime factorization result
typedef struct {
  int64_t base[MAX_FACTORS];
  int64_t n[MAX_FACTORS];
  int32_t exponent[MAX_FACTORS];
  int32_t count; // MAX_FACTORS+1 on too many factors
} factorization;

factorization *factorize(int64_t n);

// Minimal plan structure
class MinimalPlan {
  public:
  MinimalPlan(int64_t *_n, int32_t _n_dims, int32_t _region_start,
  int32_t _region_end, int32_t _flags) : n_dims(_n_dims), region_start(_region_start),
  region_end(_region_end), flags(_flags) {

    memset(base, 0, sizeof(int64_t)*MAX_REGIONS*MAX_FACTORS);
    memset(ns, 0, sizeof(int64_t)*MAX_REGIONS*MAX_FACTORS);
    memset(func, 0, sizeof(fft_func_t)*MAX_REGIONS*MAX_FACTORS);
    memset(exp, 0, sizeof(int32_t)*MAX_REGIONS*MAX_FACTORS);
    memset(num_factors, 0, sizeof(int32_t)*MAX_REGIONS);

    for (int i = 0; i < _n_dims; i++) {
      n[i] = _n[i];
    }
    gen_inner_plan();
  }

  void gen_inner_plan();

  void execute_plan(MFFTELEM **y, MFFTELEM **x, int64_t r, int64_t bp, int64_t stride);  
  bool bt_flags(int32_t flag) { return (flags & flag) != 0; };

  friend std::ostream& operator<<(std::ostream& os, const MinimalPlan& P);

  int64_t n[MAX_REGIONS]; // here: input and output size
  int32_t n_dims;         // number of dimensions
  int32_t region_start;
  int32_t region_end;
  int32_t flags;

 // parallel arrays for inner plans
  int64_t base[MAX_REGIONS][MAX_FACTORS];
  int64_t ns[MAX_REGIONS][MAX_FACTORS];
  fft_func_t func[MAX_REGIONS][MAX_FACTORS];
  int32_t exp[MAX_REGIONS][MAX_FACTORS];

  int32_t num_factors[MAX_REGIONS]; // number of factors per region
  protected:
  void add_plan_factor(int32_t r, int64_t ns, int64_t base,
                       int32_t exp, fft_func_t func);
  void plan_1d(int64_t n, int32_t rd);
};

#endif