
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

// Prime factorization result
struct factors {
  int64_t base;
  int64_t n;
  int32_t exponent;
};

std::vector<factors> factorize(int64_t n);

// Plan parameter control MinPlanConfig
struct MinimalPlanConfig {
  int64_t direct_sz = DEFAULT_DIRECT_SZ;
  int64_t small_sz = DEFAULT_SMALL_SZ;
  int64_t max_map_cache = MAX_MAP_CACHE;
};

// Minimal plan structure
class MinimalPlan {
 public:
  // _n is the dimension for each region, _n_dims number of dims
  MinimalPlan(int64_t* _n, int32_t _n_dims, int32_t _region_start, int32_t _region_end,
              int32_t _flags);

  ~MinimalPlan();

  // X preserved if not inplace, does only a single FFT in region r
  void execute_plan(MinAlignedVector& Y, MinAlignedVector& X, const int32_t region,
                    const int64_t bp, const int64_t stride) const;

  void execute_plan_no_copy(MinAlignedVector &Y, MinAlignedVector &X, const int32_t region,
                            const int64_t bp,
                            const int64_t stride) const;  // *XX may be destroyed

  // does all ffts from region_start to region_end
  void execute_multid_plan(MinAlignedVector& Y, MinAlignedVector& X, const int32_t region_start,
                           int32_t region_end, const int64_t bp,
                           const int64_t stride) const;  // X preserved if not inplace)

  void execute_multid_plan_no_copy(MinAlignedVector& Y, MinAlignedVector& X,
                                   const int32_t region_start, int32_t region_end, const int64_t bp,
                                   const int64_t stride) const;  // X preserved if not inplace)

  inline bool bt_flags(int32_t flag) { return (flags & flag) != 0; };

  friend std::ostream& operator<<(std::ostream& os, const MinimalPlan& P);

  fft_func_t* get_funcs(int r) { return regions[r].func; }
  int32_t get_region_start() { return region_start; }
  int32_t get_region_end() { return region_end; }

protected:

  // indexer needs region data
  friend void do_1d_plan(const MinimalPlan& P, MFFTELEM** YY, MFFTELEM** XX, int32_t region) noexcept;
  friend void do_1d_r0(const MinimalPlan& P, MFFTELEM **YY, MFFTELEM **XX) noexcept;
  
  // callback from indexer
  void execute_plan_no_copy(MFFTELEM **YY, MFFTELEM **XX, const int64_t region,
                            const int64_t bp,
                            const int64_t stride) const;  // *XX may be destroyed

  // callback from indexer
  void execute_multid_plan_no_copy(MFFTELEM **YY, MFFTELEM **XX,
                                   const int32_t region_start, int32_t region_end, const int64_t bp,
                                   const int64_t stride) const;  // X preserved if not inplace)

  void gen_inner_plan(int32_t flags);

  int64_t N;            // total size across all regions
  int64_t n[MAX_DIMS];  // size of each region
  int64_t strides[MAX_DIMS]; // stride of each region for multid plans
  int32_t n_dims;       // number of regions
  int32_t region_start;
  int32_t region_end;
  int32_t flags;

  struct region_data zero_region;

  struct region_data* regions = nullptr;

  void remove_two(std::vector<factors>& factors, int i, int j);
  void plan_small(std::vector<factors>& factors, int32_t region, int32_t flags);
  void add_plan_factor(int32_t r, int64_t ns, int64_t base, int32_t exp, fft_func_t func);
  void plan_1d(int64_t region_n, int32_t region, int32_t flags);
};

#endif