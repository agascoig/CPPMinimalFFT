
#include "plan.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "CPPMinimalFFT.hpp"

MinimalPlan::MinimalPlan(int64_t* _n, int32_t _n_dims, int32_t _region_start, int32_t _region_end,
                         int32_t _flags)
    : n_dims(_n_dims), region_start(_region_start), region_end(_region_end), flags(_flags) {
  minassert(n_dims <= MAX_DIMS, "Too many dimensions");
  minassert(region_end - region_start < MAX_REGIONS, "Too many regions");

  base_p = new int64_t[region_end + 1][MAX_FACTORS]();
  ns_p = new int64_t[region_end + 1][MAX_FACTORS]();
  func_p = new fft_func_t[region_end + 1][MAX_FACTORS]();
  exp_p = new int32_t[region_end + 1][MAX_FACTORS]();
  pfa_params_p = new int64_t[region_end + 1][MAX_PFA_PARAMS]();
  total_size = 1;
  for (int i = 0; i < _n_dims; i++) {
    n[i] = _n[i];
    total_size *= n[i];
  }
  gen_inner_plan(flags);
}

std::ostream& operator<<(std::ostream& os, const MinimalPlan& P) {
  os << "Plan:\n";
  os << "  n_dims: " << P.n_dims << "\n";
  os << "  region_start: " << P.region_start << "\n";
  os << "  region_end: " << P.region_end << "\n";
  os << "  flags: " << P.flags << "\n";
  for (int32_t r = P.region_start; r <= P.region_end; r++) {
    os << "  Region " << r << ": n=" << P.n[r] << " num_factors=" << P.num_factors[r] << "\n";

    const int64_t* b_p = P.base_p[r];
    const int64_t* n_p = P.ns_p[r];
    const fft_func_t* f_p = P.func_p[r];
    const int32_t* e_p = P.exp_p[r];

    for (int32_t f = 0; f < P.num_factors[r]; f++) {
      os << "    Factor " << f << ": base=" << b_p[f] << " exp=" << e_p[f] << " ns=" << n_p[f]
         << " func=" << (void*)f_p[f] << "\n";
    }
  }
  return os;
}

void MinimalPlan::add_plan_factor(int32_t region, int64_t _ns, int64_t _base, int32_t _exp,
                                  fft_func_t _func) {
  int32_t factor_idx = num_factors[region];

  minassert(region < MAX_REGIONS, "Region index out of bounds");
  minassert(num_factors[region] < MAX_FACTORS,
            "MinimalPlan::add_plan_factor Exceeded maximum factors per region");

  ns_p[region][factor_idx] = _ns;
  base_p[region][factor_idx] = _base;
  exp_p[region][factor_idx] = _exp;
  func_p[region][factor_idx] = _func;
  num_factors[region]++;
}

// Plan 1D FFT
void MinimalPlan::plan_1d(int64_t n, int32_t rd, int32_t flags) {
  if (num_factors[rd] > 0) {
    return;  // region already planned, refuse
  }
  minassert(rd < MAX_REGIONS, "Region index out of bounds");

  bool inverse = (flags & P_INVERSE) != 0;
  factorization* p_factors = factorize(n);

  bool copy_input = true;

  const int32_t factor_count = p_factors->count;
  minassert(factor_count <= MAX_FACTORS, "Too many factors to plan_1d.");

  struct sort_factor {
    int64_t n;
    int32_t index;
  };

  sort_factor factors[MAX_FACTORS];
  for (int32_t i = 0; i < factor_count; i++) {
    factors[i].n = p_factors->n[i];
    factors[i].index = i;
  }
  std::sort(factors, factors + factor_count, [](const sort_factor& a, const sort_factor& b) {
    return a.n > b.n;  // descending by n
  });

  if (n <= DIRECT_SZ) {
    add_plan_factor(rd, n, n, 1, inverse ? &direct_dft<true> : &direct_dft<false>);
    copy_input = false;
  } else if ((n & (n - 1)) == 0) {
    // Power of 2
    int32_t exp = 63 - count_leading_zeros(n);
    if ((exp % 3) == 0)
      add_plan_factor(rd, n, 8, exp / 3, inverse ? &fftr8<true> : &fftr8<false>);
    else if ((exp & 1) == 0)
      add_plan_factor(rd, n, 4, exp / 2, inverse ? &fftr4<true> : &fftr4<false>);
    else
      add_plan_factor(rd, n, 2, exp, inverse ? &fftr2<true> : &fftr2<false>);
  } else if (factor_count <= MAX_FACTORS) {
    for (int32_t j = factor_count - 1; j >= 0; j--) {
      int32_t i = factors[j].index;
      int64_t base = p_factors->base[i];
      int32_t exp = p_factors->exponent[i];
      int32_t nf = p_factors->n[i];
      fft_func_t func;
      if (nf <= DIRECT_SZ) {
        func = inverse ? &direct_dft<true> : &direct_dft<false>;
      } else {
        if ((base < DISPATCH_SZ) && (dispatch[base])) {
          func = inverse ? dispatch_inverse[base] : dispatch[base];
          copy_input = true;
        } else {
          func = inverse ? &bluestein<true> : &bluestein<false>;
        }
      }
      add_plan_factor(rd, nf, base, exp, func);
    }
  } else {
    add_plan_factor(rd, n, n, 1, inverse ? &bluestein<true> : &bluestein<false>);
    flags |= P_TOO_MANY_FACTORS;
  }

  if (copy_input && !(flags & P_INPLACE)) flags |= P_COPY_INPUT;
  free(p_factors);

  if (num_factors[rd] >= 2) generate_pfa_params(factor_count, ns_p[rd], pfa_params_p[rd]);
}

void MinimalPlan::gen_inner_plan(int32_t flags) {
  for (int64_t r = region_start; r <= region_end; r++) {
    int64_t nt = n[r];
    plan_1d(nt, r, flags);
  }
}

void MinimalPlan::execute_plan_no_copy(MFFTELEM** YY, MFFTELEM** XX, int64_t r, int64_t bp,
                                       int64_t stride) const {
  const int64_t* b_p = base_p[r];
  const int64_t* n_p = ns_p[r];
  const fft_func_t* f_p = func_p[r];
  const int32_t* e_p = exp_p[r];
  const int64_t* params_p = pfa_params_p[r];

  int64_t lf = num_factors[r];
  minassert(lf <= MAX_FACTORS, "Too many factors to execute_plan_no_copy.");
  if (!lf) return;

  switch (lf) {
    case 1:
      f_p[0](YY, XX, n_p[0], e_p[0], bp, stride, flags);
      break;
    case 2:
      prime_factor_2(YY, XX, n_p, e_p, bp, stride, flags, f_p, params_p);
      break;
    case 3:
      prime_factor_3(YY, XX, n_p, e_p, bp, stride, flags, f_p, params_p);
      break;
    case 4:
      pfa_extend_4(YY, XX, n_p, e_p, bp, stride, flags, f_p, params_p);
      break;
    case 5:
      pfa_extend_5(YY, XX, n_p, e_p, bp, stride, flags, f_p, params_p);
      break;
    case 6:
      pfa_extend_6(YY, XX, n_p, e_p, bp, stride, flags, f_p, params_p);
      break;
    default:
      minassert(0, "Too many factors, should have planned bluestein.");
  }
}

// Execute plan function with input copying if needed
void MinimalPlan::execute_plan(MinAlignedVector &Y, MinAlignedVector &X, int64_t r, int64_t bp,
                               int64_t stride) const {
  MFFTELEM *Y_data = Y.data();
  MFFTELEM *X_data = X.data();
  
  MFFTELEM** YY = &Y_data;
  MFFTELEM** XX = &X_data;
  MFFTELEM* copy_X = nullptr;
  
  execute_plan_no_copy(YY, XX, r, bp, stride);

  if (*YY != Y.data()) {
    swap(Y, X);
  }
}
