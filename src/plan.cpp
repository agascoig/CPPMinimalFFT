
#include "plan.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "CPPMinimalFFT.hpp"

struct MinimalPlanConfig MinPlanConfig;  // global struct with parameters for MinimalPlan

MinimalPlan::MinimalPlan(int64_t* _n, int32_t _n_dims, int32_t _region_start,
                         int32_t _region_end, int32_t _flags)
    : n_dims(_n_dims),
      region_start(_region_start),
      region_end(_region_end),
      flags(_flags) {
  minassert(n_dims <= MAX_DIMS, "Too many dimensions");
  minassert(region_end - region_start < MAX_REGIONS, "Too many regions");
  if (region_start == 0 && region_end == 0) {
    regions = &zero_region;
  } else {
    regions = new struct region_data[region_end - region_start + 1]();
  }
  N = 1;
  for (int i = 0; i < _n_dims; i++) {
    strides[i] = N;
    int64_t ni = _n[i];
    n[i] = ni;
    N *= ni;
    if (i >= region_start && i <= region_end) {
      regions[i - region_start].n = _n[i];
    }
  }
  gen_inner_plan(flags);
}

MinimalPlan::~MinimalPlan() {
  for (int i = 0; i < region_end - region_start + 1; ++i) {
    if (regions[i].nm != nullptr) delete[] regions[i].nm;
    if (regions[i].km != nullptr) delete[] regions[i].km;
  }
  if (regions != &zero_region) delete[] regions;
}

std::ostream& operator<<(std::ostream& os, const MinimalPlan& P) {
  os << "Plan: N=" << P.N << "\n";
  os << "  n_dims: " << P.n_dims << "\n  n=";
  for (int i = 0; i < P.n_dims; i++) {
    os << P.n[i] << " ";
  }
  os << "\n  strides=";
  for (int i = 0; i < P.n_dims; i++) {
    os << P.strides[i] << " ";
  }
  os << "\n  region_start: " << P.region_start << "\n";
  os << "  region_end: " << P.region_end << "\n";
  os << "  flags: " << P.flags << "\n";

  for (int32_t i = P.region_start; i <= P.region_end; i++) {
    const auto& rd = P.regions[i - P.region_start];
    os << "  Region " << i << ": n=" << rd.n << " num_factors=" << rd.num_factors
       << "\n";

    os << rd << std::endl;
  }
    return os;
}

struct small_sruct {
  int32_t N, b1, e1, b2, e2;
} small_tbl[] = {
    {28, 2, 2, 7, 1}, {26, 2, 1, 13, 1}, {24, 2, 3, 3, 1}, {21, 3, 1, 7, 1},
    {20, 2, 2, 5, 1}, {19, 19, 1, 0, 0}, {18, 2, 1, 9, 1}, {17, 17, 1, 0, 0},
    {16, 2, 4, 0, 0}, {15, 3, 1, 5, 1},  {14, 2, 1, 7, 1}, {13, 13, 1, 0, 0},
    {12, 2, 2, 3, 1}, {11, 11, 1, 0, 0}, {10, 2, 1, 5, 1}, {9, 3, 2, 0, 0},
    {8, 2, 3, 0, 0},  {7, 7, 1, 0, 0},   {6, 2, 1, 3, 1},  {5, 5, 1, 0, 0},
    {4, 2, 2, 0, 0},  {3, 3, 1, 0, 0},   {2, 2, 1, 0, 0}};
static const int SMALL_TBL_SZ = sizeof(small_tbl) / sizeof(small_sruct);

void MinimalPlan::remove_two(std::vector<factors>& factors, int i, int j) {
  if (i == j) return;
  if (i > j) std::swap(i, j);
  size_t last = factors.size() - 1;
  size_t second_last = factors.size() - 2;
  if (j != last) factors[j] = std::move(factors[last]);
  if (i != second_last) {
    // second_last may have just moved to j
    size_t src = second_last;
    if (j == second_last) src = last;
    factors[i] = std::move(factors[src]);
  }
  factors.pop_back();
  factors.pop_back();
}

void MinimalPlan::plan_small(std::vector<factors>& factors, int32_t region,
                             int32_t flags) {
  const bool inverse = (flags & P_INVERSE) != 0;
  int f1, f2;
  for (int32_t i = 0; i < SMALL_TBL_SZ; i++) {
    const int32_t nf = factors.size();
    f1 = -1;
    f2 = -1;
    fft_func_t fn = get_small_func(small_tbl[i].N, inverse);
    if ((fn == nullptr) || (small_tbl[i].N > MinPlanConfig.small_sz)) {
      continue;
    }
    bool two_factors = (small_tbl[i].b2 != 0);
    for (int32_t j = 0; j < nf; j++) {
      if (factors[j].base == small_tbl[i].b1 &&
          factors[j].exponent == small_tbl[i].e1) {
        f1 = j;
        break;
      }
    }
    for (int32_t j = 0; j < nf; j++) {
      if (j != f1 && factors[j].base == small_tbl[i].b2 &&
          factors[j].exponent == small_tbl[i].e2) {
        f2 = j;
        break;
      }
    }
    if (!two_factors && f1 != -1) {
      int32_t b = factors[f1].n;
      add_plan_factor(region, b, b, 1, fn);
      factors[f1] = std::move(factors.back());
      factors.pop_back();
      if (factors.empty()) return;
      i = -1;
    } else if (two_factors && f1 != -1 && f2 != -1) {
      int32_t b = factors[f1].n * factors[f2].n;
      add_plan_factor(region, b, b, 1, fn);
      remove_two(factors, f1, f2);
      if (factors.empty()) return;
      i = -1;
    }
  }
}

void MinimalPlan::add_plan_factor(int32_t region, int64_t _ns, int64_t _base,
                                  int32_t _exp, fft_func_t _func) {
  minassert(region <= region_end, "Region index out of bounds");
  int32_t factor_idx = regions[region - region_start].num_factors;
  minassert(factor_idx < MAX_FACTORS,
            "MinimalPlan::add_plan_factor Exceeded maximum factors per region");
  auto& r = regions[region - region_start];
  r.ns[factor_idx] = _ns;
  r.base[factor_idx] = _base;
  r.exp[factor_idx] = _exp;
  r.func[factor_idx] = _func;
  r.num_factors++;
}

// Plan 1D FFT
void MinimalPlan::plan_1d(int64_t region_n, int32_t region, int32_t flags) {
  minassert(region < MAX_REGIONS, "Region index out of bounds");
  auto& rd = regions[region - region_start];
  if (rd.num_factors > 0) {
    return;  // region already planned, refuse
  }
  bool inverse = (flags & P_INVERSE) != 0;
  std::vector<factors> factors = factorize(region_n);
  plan_small(factors, region, flags);  // do small block sizes first
  bool copy_input = true;
  if (!factors.empty()) {
    struct sort_factor {
      int64_t n;
      int32_t index;
      bool bluestein;
    };
    sort_factor sorted_factors[MAX_FACTORS];
    for (int32_t i = 0; i < factors.size(); i++) {
      sorted_factors[i].n = factors[i].n;
      sorted_factors[i].index = i;
      sorted_factors[i].bluestein = (factors[i].n > MinPlanConfig.direct_sz &&
                                     (factors[i].base >= DISPATCH_SZ ||
                                      dispatch[factors[i].base] == nullptr))
                                        ? true
                                        : false;
    }
    std::sort(
        sorted_factors, sorted_factors + factors.size(),
        [](const sort_factor& a, const sort_factor& b) {
          if (a.bluestein != b.bluestein)
            return a
                .bluestein;  // give bluestein priority (single-precision SIMD)
          return a.n > b.n;  // descending by n
        });
    if (region_n <= MinPlanConfig.direct_sz) {
      add_plan_factor(region, region_n, region_n, 1,
                      inverse ? &direct_dft<true> : &direct_dft<false>);
      copy_input = false;
    } else if ((region_n & (region_n - 1)) == 0) {
      // Power of 2
      int32_t exp = 63 - count_leading_zeros(region_n);
      if ((exp % 4) == 0)
        add_plan_factor(region, region_n, 16, exp / 4,
                        inverse ? &fftr16<true> : &fftr16<false>);
      else if ((exp % 3) == 0)
        add_plan_factor(region, region_n, 8, exp / 3,
                        inverse ? &fftr8<true> : &fftr8<false>);
      else if ((exp % 2) == 0)
        add_plan_factor(region, region_n, 4, exp / 2,
                        inverse ? &fftr4<true> : &fftr4<false>);
      else
        add_plan_factor(region, region_n, 2, exp,
                        inverse ? &fftr2<true> : &fftr2<false>);
    } else if (factors.size() <= MAX_FACTORS) {
      for (int32_t j = 0; j < factors.size(); j++) {
        int32_t i = sorted_factors[j].index;
        int64_t base = factors[i].base;
        int32_t exp = factors[i].exponent;
        int32_t fn = factors[i].n;
        fft_func_t func;
        if (fn <= MinPlanConfig.direct_sz) {
          func = inverse ? &direct_dft<true> : &direct_dft<false>;
        } else {
          if ((base == 3) && ((exp & 1) == 0)) {
            // promot to fftr9
            base = 9;
            exp /= 2;
          }
          if ((base < DISPATCH_SZ) && (dispatch[base])) {
            func = inverse ? dispatch_inverse[base] : dispatch[base];
            copy_input = true;
          } else {
            func = inverse ? &bluestein<true> : &bluestein<false>;
          }
        }
        add_plan_factor(region, fn, base, exp, func);
      }
    } else {
      add_plan_factor(region, region_n, region_n, 1,
                      inverse ? &bluestein<true> : &bluestein<false>);
      flags |= P_TOO_MANY_FACTORS;
    }
    if (copy_input && !(flags & P_INPLACE)) flags |= P_COPY_INPUT;
  }
  if (rd.num_factors >= 2) {
    QPs(rd.QPs, rd.num_factors, rd.ns);
    if (N <= MinPlanConfig.max_map_cache) {
      rd.nm = generate_nmap(rd.num_factors, N, rd.ns, rd.QPs);
      rd.km = generate_kmap(rd.num_factors, N, rd.ns, rd.QPs);
    }
  }
  create_region_data_strides(rd);
}

void MinimalPlan::gen_inner_plan(int32_t flags) {
  for (int64_t r = region_start; r <= region_end; r++) {
    int64_t nt = regions[r - region_start].n;
    plan_1d(nt, r, flags);
  }
}

void MinimalPlan::execute_multid_plan_no_copy(MFFTELEM** YY, MFFTELEM** XX,
                                              const int32_t region_start,
                                              const int32_t region_end,
                                              const int64_t bp,
                                              const int64_t stride) const {
  int64_t Ns[MAX_DIMS] = {0};
  int i;
  for (i = 0; i <= region_end - region_start; ++i) {
    if ((i & 1) == 0)
      do_fft_planned(*this, YY, XX, region_start + i);
    else
      do_fft_planned(*this, XX, YY, region_start + i);
  }
  if ((i & 1) == 0) {
    MFFTELEM* tmp = *XX;
    *YY = *XX;
    *XX = tmp;
  }
}

void MinimalPlan::execute_multid_plan_no_copy(
    MinAlignedVector& Y, MinAlignedVector& X, const int32_t region_start,
    const int32_t region_end, const int64_t bp, const int64_t stride) const {
  MFFTELEM* Y_data = Y.data();
  MFFTELEM* X_data = X.data();
  MFFTELEM** YY = &Y_data;
  MFFTELEM** XX = &X_data;
  execute_multid_plan_no_copy(YY, XX, region_start, region_end, bp, stride);
  if (*YY != Y.data()) {
    swap(Y, X);
  }
  if (flags & P_INPLACE) X = Y;
}

void MinimalPlan::execute_multid_plan(MinAlignedVector& Y, MinAlignedVector& X,
                                      const int32_t region_start,
                                      const int32_t region_end,
                                      const int64_t bp,
                                      const int64_t stride) const {
  MinAlignedVector copy_X(X);
  execute_multid_plan_no_copy(Y, X, region_start, region_end, bp, stride);
  if (X != copy_X) X = copy_X;
}

void MinimalPlan::execute_plan_no_copy(MFFTELEM** YY, MFFTELEM** XX,
                                       const int64_t region, const int64_t bp,
                                       const int64_t stride) const {
  const auto& rd = regions[region - region_start];
  const int32_t nf = rd.num_factors;
  minassert(nf <= MAX_FACTORS, "Too many factors to execute_plan_no_copy.");
  if (nf == 1) {
    rd.func[0](YY, XX, rd.ns[0], rd.exp[0], bp, stride, flags);
    return;
  }
  prime_factor(YY, XX, rd, bp, stride, flags, nf);
  // *YY and *XX may have flipped
}

void MinimalPlan::execute_plan_no_copy(MinAlignedVector& Y, MinAlignedVector& X,
                                       const int32_t region, const int64_t bp,
                                       const int64_t stride) const {
  MFFTELEM* Y_data = Y.data();
  MFFTELEM* X_data = X.data();
  MFFTELEM** YY = &Y_data;
  MFFTELEM** XX = &X_data;
  execute_plan_no_copy(YY, XX, region, bp, stride);
  if (*YY != Y.data()) {
    swap(Y, X);
  }
  if (flags & P_INPLACE) X = Y;
}

// Execute plan function with input copying if needed
void MinimalPlan::execute_plan(MinAlignedVector& Y, MinAlignedVector& X,
                               const int32_t region, const int64_t bp,
                               const int64_t stride) const {
  MinAlignedVector copy_X(X);
  execute_plan_no_copy(Y, X, region, bp, stride);
  if (X != copy_X) X = copy_X;
}
