
#include "plan.hpp"

#include <algorithm>
#include <cassert>
#include <cstdbool>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "CPPMinimalFFT.hpp"

std::ostream &operator<<(std::ostream &os, const MinimalPlan &P) {
  os << "Plan:\n";
  os << "  n_dims: " << P.n_dims << "\n";
  os << "  region_start: " << P.region_start << "\n";
  os << "  region_end: " << P.region_end << "\n";
  os << "  flags: " << P.flags << "\n";
  for (int32_t r = P.region_start; r <= P.region_end; r++) {
    os << "  Region " << r << ": n=" << P.n[r]
       << " num_factors=" << P.num_factors[r] << "\n";

    const int64_t *b_p = P.base_p[r];
    const int64_t *n_p = P.ns_p[r];
    const fft_func_t *f_p = P.func_p[r];
    const int32_t *e_p = P.exp_p[r];

    for (int32_t f = 0; f < P.num_factors[r]; f++) {
      os << "    Factor " << f << ": base=" << b_p[f] << " exp=" << e_p[f]
         << " ns=" << n_p[f] << " func=" << (void *)f_p[f] << "\n";
    }
  }
  return os;
}

void MinimalPlan::add_plan_factor(int32_t region, int64_t _ns, int64_t _base,
                                  int32_t _exp, fft_func_t _func) {
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
void MinimalPlan::plan_1d(int64_t n, int32_t rd) {
  if (num_factors[rd] > 0) {
    return;  // region already planned
  }
  minassert(rd < MAX_REGIONS, "Region index out of bounds");

  factorization *p_factors = factorize(n);

  bool copy_input = true;

  const int32_t factor_count = p_factors->count;

  struct sort_factor {
    int64_t n;
    int32_t index;
  };

  sort_factor factors[MAX_FACTORS];
  for (int32_t i = 0; i < factor_count; i++) {
    factors[i].n = p_factors->n[i];
    factors[i].index = i;
  }
  std::sort(factors, factors + factor_count,
            [](const sort_factor &a, const sort_factor &b) {
              return a.n > b.n;  // descending by n
            });

  if (n <= DIRECT_SZ) {
    add_plan_factor(rd, n, n, 1, &direct_dft);
    copy_input = false;
  } else if ((n & (n - 1)) == 0) {
    // Power of 2
    int32_t exp = 63 - count_leading_zeros(n);
    add_plan_factor(rd, n, 2, exp, &fftr2);
  } else if (factor_count <= MAX_FACTORS) {
    for (int32_t j = factor_count - 1; j >= 0; j--) {
      int32_t i = factors[j].index;
      int64_t base = p_factors->base[i];
      int32_t exp = p_factors->exponent[i];
      int32_t nf = p_factors->n[i];
      fft_func_t func;
      if (nf <= DIRECT_SZ) {
        func = &direct_dft;
      } else {
        if ((base < DISPATCH_SZ) && (dispatch[base])) {
          func = dispatch[base];
          copy_input = true;
        } else {
          func = &bluestein;
        }
      }
      add_plan_factor(rd, nf, base, exp, func);
    }
  } else {
    add_plan_factor(rd, n, n, 1, &bluestein);
    flags |= P_TOO_MANY_FACTORS;
  }

  if (copy_input && !(flags & P_INPLACE)) flags |= P_COPY_INPUT;
  free(p_factors);

  if (num_factors[rd] >= 2)
    generate_pfa_params(factor_count, ns_p[rd], pfa_params_p[rd]);
}

void MinimalPlan::gen_inner_plan() {
  for (int64_t r = region_start; r <= region_end; r++) {
    int64_t nt = n[r];
    plan_1d(nt, r);
  }
}

void MinimalPlan::execute_plan_no_copy(MFFTELEM **YY, MFFTELEM **XX, int64_t r,
                                       int64_t bp, int64_t stride) const {
  const int64_t *b_p = base_p[r];
  const int64_t *n_p = ns_p[r];
  const fft_func_t *f_p = func_p[r];
  const int32_t *e_p = exp_p[r];
  const int64_t *params_p = pfa_params_p[r];

  int64_t lf = num_factors[r];
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
void MinimalPlan::execute_plan(MFFTELEM *Y, MFFTELEM *X, int64_t r, int64_t bp,
                               int64_t stride) const {
  MFFTELEM *orig_Y = Y;
  MFFTELEM **YY = &Y;
  MFFTELEM **XX = &X;
  MFFTELEM *copy_X = nullptr;

  minassert(
      Y != X || (flags & P_INPLACE),
      "Input and output buffers must not be the same unless in-place plan.");

  bool copy_input = ((flags & P_COPY_INPUT) != 0);

  if (copy_input) {
    copy_X = (MFFTELEM *)minaligned_alloc(sizeof(MFFTELEM), sizeof(MFFTELEM),
                                          total_size);
    memcpy(copy_X, X, total_size * sizeof(MFFTELEM));
    *XX = copy_X;
  }

  execute_plan_no_copy(YY, XX, r, bp, stride);

  if (*YY != orig_Y)
    memcpy(
        orig_Y, *YY,
        total_size * sizeof(MFFTELEM));  // swapped, so copy to original output
  else if (flags & P_INPLACE)
    memcpy(X, *YY,
           total_size * sizeof(MFFTELEM));  // in-place, so copy back to
                                            // original input if not already
  if (copy_X) free(copy_X);
}
