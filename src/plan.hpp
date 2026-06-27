
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

static const int DEFAULT_DIRECT_SZ = 15;
static const int DEFAULT_SMALL_SZ = 28;
static const int MAX_PFA_PARAMS = (2 * (MAX_FACTORS - 1));

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
  // _n is the dimension for each region, _n_dims number of dims
  MinimalPlan(int64_t* _n, int32_t _n_dims, int32_t _region_start, int32_t _region_end,
              int32_t _flags, int64_t _direct_sz=DEFAULT_DIRECT_SZ, int64_t _small_sz=DEFAULT_SMALL_SZ);

  ~MinimalPlan();

  void execute_plan_no_copy(MFFTELEM** YY, MFFTELEM** XX, int64_t r, int64_t bp,
                            int64_t stride) const;  // *XX may be destroyed

  // does all ffts from region_start to region_end
  void execute_multid_plan(MinAlignedVector& Y, MinAlignedVector& X, int32_t region_start,
                           int32_t region_end, int64_t bp,
                           int64_t stride) const;  // X preserved if not inplace)

  // X preserved if not inplace, does only a single FFT in region r
  void execute_plan(
      MinAlignedVector& Y, MinAlignedVector& X, int32_t r, int64_t bp,
      int64_t stride) const;
  inline bool bt_flags(int32_t flag) { return (flags & flag) != 0; };

  friend std::ostream& operator<<(std::ostream& os, const MinimalPlan& P);

  fft_func_t* get_funcs(int r) { return pfunc[r]; }
  int32_t get_region_start() { return region_start; }
  int32_t get_region_end() { return region_end; }

 protected:
  void gen_inner_plan(int32_t flags);

  int64_t N;
  int64_t n[MAX_REGIONS];  // here: input and output size
  int32_t n_dims;          // number of dimensions
  int32_t region_start;
  int32_t region_end;
  int32_t flags;
  int64_t direct_sz;
  int64_t small_sz;

  // pointers to regions
  int64_t pbase[MAX_REGIONS][MAX_FACTORS];
  int64_t pns[MAX_REGIONS][MAX_FACTORS];
  fft_func_t pfunc[MAX_REGIONS][MAX_FACTORS];
  int32_t pexp[MAX_REGIONS][MAX_FACTORS];

  int64_t* QPs_p[MAX_REGIONS] = {nullptr};
  MAP_CACHE_T* nm_p[MAX_REGIONS] = {nullptr};
  MAP_CACHE_T* km_p[MAX_REGIONS] = {nullptr};

  int32_t num_factors[MAX_REGIONS] = {0};  // zero init number of factors per region

  void add_plan_factor(int32_t r, int64_t ns, int64_t base, int32_t exp, fft_func_t func);
  void plan_1d(int64_t n, int32_t rd, int32_t flags);
};

#endif