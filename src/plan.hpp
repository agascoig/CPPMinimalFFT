
#ifndef __PLAN_H__
#define __PLAN_H__

#include <cstdint>

#include "CPPMinimalFFT.hpp"

static const int P_NONE = 0;
static const int P_INVERSE = 1;
static const int P_INPLACE = 2;
static const int P_REAL = 4;
static const int P_ISBFFT = 8;
static const int P_ODD = 16;
static const int P_SCALED = 32;
static const int P_TOO_MANY_FACTORS = 64;
static const int P_COPY_INPUT = 128;

static const int DIRECT_SZ = 15;
static const int MAX_FACTORS = 6;
static const int MAX_PFA_PARAMS = 10;

// Prime factorization result
typedef struct {
  int64_t base[MAX_FACTORS];
  int64_t n[MAX_FACTORS];
  int32_t exponent[MAX_FACTORS];
  int32_t count;  // MAX_FACTORS+1 on too many factors
} factorization;

factorization *factorize(int64_t n);

// Minimal plan structure
class MinimalPlan {
 public:
  MinimalPlan(int64_t *_n, int32_t _n_dims, int32_t _region_start,
              int32_t _region_end, int32_t _flags)
      : n_dims(_n_dims),
        region_start(_region_start),
        region_end(_region_end),
        flags(_flags) {
    minassert(n_dims <= MAX_DIMS, "Too many dimensions");
    minassert(region_end - region_start < MAX_REGIONS, "Too many regions");

    if (region_end != 0) {
      base_p = new int64_t[region_end + 1][MAX_FACTORS]();
      ns_p = new int64_t[region_end + 1][MAX_FACTORS]();
      func_p = new fft_func_t[region_end + 1][MAX_FACTORS]();
      exp_p = new int32_t[region_end + 1][MAX_FACTORS]();
      pfa_params_p = new int64_t[region_end + 1][MAX_PFA_PARAMS]();
    } else {  // only one region
      std::fill_n(base, MAX_FACTORS, 0);
      std::fill_n(ns, MAX_FACTORS, 0);
      std::fill_n(func, MAX_FACTORS, nullptr);
      std::fill_n(exp, MAX_FACTORS, 0);
      std::fill_n(pfa_params, MAX_PFA_PARAMS, 0);

      base_p = (int64_t (*)[MAX_FACTORS])base;
      ns_p = (int64_t (*)[MAX_FACTORS])ns;
      func_p = (fft_func_t(*)[MAX_FACTORS])func;
      exp_p = (int32_t (*)[MAX_FACTORS])exp;
      pfa_params_p = (int64_t (*)[MAX_PFA_PARAMS])pfa_params;
    }
    std::fill_n(num_factors, MAX_REGIONS, 0);

    total_size = 1;
    for (int i = 0; i < _n_dims; i++) {
      n[i] = _n[i];
      total_size *= n[i];
    }
    gen_inner_plan();
  }

  ~MinimalPlan() {
    if (base_p != (int64_t (*)[MAX_FACTORS])base) {
      delete[] base_p;
      delete[] ns_p;
      delete[] func_p;
      delete[] exp_p;
      delete[] pfa_params_p;
    }
  }

  void execute_plan_no_copy(MFFTELEM **YY, MFFTELEM **XX, int64_t r, int64_t bp,
                            int64_t stride) const;  // *XX may be destroyed

  void execute_plan(MFFTELEM *Y, MFFTELEM *X, int64_t r, int64_t bp,
                    int64_t stride) const;  // X preserved if not inplace
  inline bool bt_flags(int32_t flag) { return (flags & flag) != 0; };

  friend std::ostream &operator<<(std::ostream &os, const MinimalPlan &P);

  fft_func_t *get_funcs(int r) { return func_p[r]; }

 protected:
  void gen_inner_plan();

  int64_t total_size;
  int64_t n[MAX_REGIONS];  // here: input and output size
  int32_t n_dims;          // number of dimensions
  int32_t region_start;
  int32_t region_end;
  int32_t flags;

  // parallel arrays for region_end=0 plans
  int64_t base[MAX_FACTORS];
  int64_t ns[MAX_FACTORS];
  fft_func_t func[MAX_FACTORS];
  int32_t exp[MAX_FACTORS];

  // pfa parameters
  int64_t pfa_params[MAX_PFA_PARAMS];

  // pointers to regions
  int64_t (*base_p)[MAX_FACTORS];
  int64_t (*ns_p)[MAX_FACTORS];
  fft_func_t (*func_p)[MAX_FACTORS];
  int32_t (*exp_p)[MAX_FACTORS];

  // pointer to prime factor parameters
  int64_t (*pfa_params_p)[MAX_PFA_PARAMS];

  int32_t num_factors[MAX_REGIONS];  // number of factors per region

  void add_plan_factor(int32_t r, int64_t ns, int64_t base, int32_t exp,
                       fft_func_t func);
  void plan_1d(int64_t n, int32_t rd);
};

#endif