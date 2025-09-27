
#include "plan.hpp"
#include "CPPMinimalFFT.hpp"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

std::ostream &operator<<(std::ostream &os, const MinimalPlan &P) {
  os << "Plan:\n";
  os << "  n_dims: " << P.n_dims << "\n";
  os << "  region_start: " << P.region_start << "\n";
  os << "  region_end: " << P.region_end << "\n";
  os << "  flags: " << P.flags << "\n";
  for (int32_t r = P.region_start; r <= P.region_end; r++) {
    os << "  Region " << r << ": n=" << P.n[r]
       << " num_factors=" << P.num_factors[r] << "\n";

    const int64_t *base_p = P.base[r];
    const int64_t *ns_p = P.ns[r];
    const fft_func_t *func_p = P.func[r];
    const int32_t *exp_p = P.exp[r];

    for (int32_t f = 0; f < P.num_factors[r]; f++) {
      os << "    Factor " << f << ": base=" << base_p[f] << " exp=" << exp_p[f]
         << " ns=" << ns_p[f] << " func=" << (void *)func_p[f] << "\n";
    }
  }
  return os;
}

void MinimalPlan::add_plan_factor(int32_t region, int64_t _ns, int64_t _base,
                                  int32_t _exp, fft_func_t _func) {
  int32_t factor_idx = num_factors[region];

  minassert(num_factors[region] < MAX_FACTORS,
            "MinimalPlan::add_plan_factor Exceeded maximum factors per region");

  ns[region][factor_idx] = _ns;
  base[region][factor_idx] = _base;
  exp[region][factor_idx] = _exp;
  func[region][factor_idx] = _func;

  num_factors[region]++;
}

// Plan 1D FFT
void MinimalPlan::plan_1d(int64_t n, int32_t rd) {
  if (num_factors[rd] > 0) {
    return; // region already planned
  }
  minassert(rd < MAX_REGIONS, "Region index out of bounds");

  factorization *p_factors = factorize(n);

  if (n <= DIRECT_SZ) {
    add_plan_factor(rd, n, n, 1, &direct_dft);
  } else if ((n & (n - 1)) == 0) {
    // Power of 2
    int32_t exp = 63 - count_leading_zeros(n);
    add_plan_factor(rd, n, 2, exp, &fftr2);
  } else if (p_factors->count <= MAX_FACTORS) {

    for (int32_t i = p_factors->count - 1; i >= 0; i--) {
      int64_t base = p_factors->base[i];
      int32_t exp = p_factors->exponent[i];
      int32_t nf = p_factors->n[i];
      fft_func_t func = nullptr;
      if (nf <= DIRECT_SZ) {
        func = &direct_dft;
      } else {
        func = &bluestein;
        if ((base < dispatch_sz) && (dispatch[base])) {
          func = dispatch[base];
        }
      }
      add_plan_factor(rd, nf, base, exp, func);
    }
  } else {
    add_plan_factor(rd, n, n, 1, &bluestein);
    flags |= P_TOO_MANY_FACTORS;
  }

  free(p_factors);
}

void MinimalPlan::gen_inner_plan() {
  for (int64_t r = region_start; r <= region_end; r++) {
    int64_t nt = n[r];
    plan_1d(nt, r);
  }
}

// Execute plan function
void MinimalPlan::execute_plan(MFFTELEM **YY, MFFTELEM **XX, int64_t r, int64_t bp,
                  int64_t stride) {
  int64_t lf = num_factors[r];
  if (!lf)
      return;

  int64_t *base_p = base[r];
  int64_t *ns_p = ns[r];
  fft_func_t *func_p = func[r];
  int32_t *es_p = exp[r];

  switch(lf) {
    case 1:
    func_p[0](YY, XX, ns_p[0], es_p[0], bp, stride, flags);
    break;
    case 2:
    prime_factor_2(YY, XX, es_p, ns_p, func_p, bp, stride, flags);
    break;
    case 3:
    prime_factor_3(YY, XX, es_p, ns_p, func_p, bp, stride, flags);
    break;
    case 4:
    pfa_extend_4(YY, XX, es_p, ns_p, func_p, bp, stride, flags);
    break;
    case 5:
    pfa_extend_5(YY, XX, es_p, ns_p, func_p, bp, stride, flags);
    break;
    case 6:
    pfa_extend_6(YY, XX, es_p, ns_p, func_p, bp, stride, flags);
    break;
    default:
    minassert(0, "Too many factors, should have planned bluestein.");
  }
}
