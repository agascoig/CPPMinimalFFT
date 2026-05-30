
#ifndef __PLAN_H__
#define __PLAN_H__

#include <cstdint>
#include <vector>

#include "CPPMinimalFFT.hpp"
#include "pfa.hpp"

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
static const int MAX_PFA_PARAMS = (2*(MAX_FACTORS-1));

// Prime factorization result
typedef struct {
  int64_t base[MAX_FACTORS];
  int64_t n[MAX_FACTORS];
  int32_t exponent[MAX_FACTORS];
  int32_t count;  // MAX_FACTORS+1 on too many factors
} factorization;

factorization* factorize(int64_t n);

// Minimal plan structure
class MinimalPlan {
 public:
  MinimalPlan(int64_t* _n, int32_t _n_dims, int32_t _region_start, int32_t _region_end,
              int32_t _flags);

  ~MinimalPlan() {
    delete[] base_p;
    delete[] ns_p;
    delete[] func_p;
    delete[] exp_p;
    for (int i=0;i<MAX_REGIONS;++i) {
      if (QPs_p[i]!=nullptr)
         delete QPs_p[i];
      if (nm_p[i]!=nullptr)
         delete nm_p[i];
      if (km_p[i]!=nullptr)
         delete km_p[i];
    }
  }

  void execute_plan_no_copy(MFFTELEM** YY, MFFTELEM** XX, int64_t r, int64_t bp,
                            int64_t stride) const;  // *XX may be destroyed
  void execute_plan(MinAlignedVector& Y, MinAlignedVector& X, int64_t r, int64_t bp,
                    int64_t stride) const;  // X preserved if not inplace
  inline bool bt_flags(int32_t flag) { return (flags & flag) != 0; };

  friend std::ostream& operator<<(std::ostream& os, const MinimalPlan& P);

  fft_func_t* get_funcs(int r) { return func_p[r]; }

 protected:
  void gen_inner_plan(int32_t flags);

  int64_t N;
  int64_t n[MAX_REGIONS];  // here: input and output size
  int32_t n_dims;          // number of dimensions
  int32_t region_start;
  int32_t region_end;
  int32_t flags;

  // pointers to regions
  int64_t (*base_p)[MAX_FACTORS] = {nullptr};
  int64_t (*ns_p)[MAX_FACTORS] = {nullptr};
  fft_func_t (*func_p)[MAX_FACTORS] = {nullptr};
  int32_t (*exp_p)[MAX_FACTORS] = {nullptr};

  int64_t *QPs_p[MAX_REGIONS] = {nullptr};
  MAP_CACHE_T *nm_p[MAX_REGIONS] = {nullptr};
  MAP_CACHE_T *km_p[MAX_REGIONS] = {nullptr};

  int32_t num_factors[MAX_REGIONS] = {0};  // zero init number of factors per region

  void add_plan_factor(int32_t r, int64_t ns, int64_t base, int32_t exp, fft_func_t func);
  void plan_1d(int64_t n, int32_t rd, int32_t flags);
};

#endif