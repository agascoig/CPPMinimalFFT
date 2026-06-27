
#include "plan.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "CPPMinimalFFT.hpp"

MinimalPlan::MinimalPlan(int64_t* _n, int32_t _n_dims, int32_t _region_start, int32_t _region_end,
                         int32_t _flags, int64_t _direct_sz, int64_t _small_sz)
    : n_dims(_n_dims),
      region_start(_region_start),
      region_end(_region_end),
      flags(_flags),
      direct_sz(_direct_sz),
      small_sz(_small_sz) {
  minassert(n_dims <= MAX_DIMS, "Too many dimensions");
  minassert(region_end - region_start < MAX_REGIONS, "Too many regions");

  N = 1;
  for (int i = 0; i < _n_dims; i++) {
    n[i] = _n[i];
    N *= n[i];
  }
  for (int i = 0; i <= region_end; i++) {
    QPs_p[i] = new int64_t[MAX_FACTORS];
  }
  gen_inner_plan(flags);
}

MinimalPlan::~MinimalPlan() {
  for (int i = 0; i < MAX_REGIONS; ++i) {
    if (QPs_p[i] != nullptr) delete[] QPs_p[i];
    if (nm_p[i] != nullptr) delete[] nm_p[i];
    if (km_p[i] != nullptr) delete[] km_p[i];
  }
}

std::ostream& operator<<(std::ostream& os, const MinimalPlan& P) {
  os << "Plan:\n";
  os << "  n_dims: " << P.n_dims << "\n";
  os << "  region_start: " << P.region_start << "\n";
  os << "  region_end: " << P.region_end << "\n";
  os << "  flags: " << P.flags << "\n";
  for (int32_t r = P.region_start; r <= P.region_end; r++) {
    os << "  Region " << r << ": n=" << P.n[r] << " num_factors=" << P.num_factors[r] << "\n";

    const int64_t* b_p = P.pbase[r];
    const int64_t* n_p = P.pns[r];
    const fft_func_t* f_p = P.pfunc[r];
    const int32_t* e_p = P.pexp[r];

    for (int32_t f = 0; f < P.num_factors[r]; f++) {
      os << "    Factor " << f << ": base=" << b_p[f] << " exp=" << e_p[f] << " ns=" << n_p[f];
      // << " func=" << (void*)f_p[f]
      fft_func_t func = f_p[f];
      for (int j = 0; j < sizeof(fns_names) / sizeof(fn_name_s); ++j) {
        if (func == fns_names[j].fn) {
          os << " func = " << fns_names[j].name;
          break;
        }
      }
      os << "\n";
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

  pns[region][factor_idx] = _ns;
  pbase[region][factor_idx] = _base;
  pexp[region][factor_idx] = _exp;
  pfunc[region][factor_idx] = _func;
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

  const int32_t nf = p_factors->count;
  minassert(nf <= MAX_FACTORS, "Too many factors to plan_1d.");

  struct sort_factor {
    int64_t n;
    int32_t index;
    bool bluestein;
  };

  sort_factor factors[MAX_FACTORS];
  for (int32_t i = 0; i < nf; i++) {
    factors[i].n = p_factors->n[i];
    factors[i].index = i;
    factors[i].bluestein =
        (p_factors->n[i] > direct_sz &&
         (p_factors->base[i] >= DISPATCH_SZ || dispatch[p_factors->base[i]] == nullptr))
            ? true
            : false;
  }
  std::sort(factors, factors + nf, [](const sort_factor& a, const sort_factor& b) {
    if (a.bluestein != b.bluestein)
      return a.bluestein;  // give bluestein priority (single-precision SIMD)
    return a.n > b.n;      // descending by n
  });

  if (n <= direct_sz) {
    add_plan_factor(rd, n, n, 1, inverse ? &direct_dft<true> : &direct_dft<false>);
    copy_input = false;
  } else if ((n & (n - 1)) == 0) {
    // Power of 2
    int32_t exp = 63 - count_leading_zeros(n);
    if ((exp % 7) == 0)
      add_plan_factor(rd, n, 16, exp / 4, inverse ? &fftr16<true> : &fftr16<false>);
    else if ((exp % 3) == 0)
      add_plan_factor(rd, n, 8, exp / 3, inverse ? &fftr8<true> : &fftr8<false>);
    else if ((exp & 1) == 0)
      add_plan_factor(rd, n, 4, exp / 2, inverse ? &fftr4<true> : &fftr4<false>);
    else
      add_plan_factor(rd, n, 2, exp, inverse ? &fftr2<true> : &fftr2<false>);
  } else if (nf <= MAX_FACTORS) {
    for (int32_t j = 0; j < nf; j++) {
      int32_t i = factors[j].index;
      int64_t base = p_factors->base[i];
      int32_t exp = p_factors->exponent[i];
      int32_t nf = p_factors->n[i];
      fft_func_t func;
      if (nf <= direct_sz) {
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
      add_plan_factor(rd, nf, base, exp, func);
    }
  } else {
    add_plan_factor(rd, n, n, 1, inverse ? &bluestein<true> : &bluestein<false>);
    flags |= P_TOO_MANY_FACTORS;
  }

  if (copy_input && !(flags & P_INPLACE)) flags |= P_COPY_INPUT;
  free(p_factors);

  if (num_factors[rd] >= 2) {
    QPs_p[rd] = generate_QPs(nf, pns[rd]);
    nm_p[rd] = generate_nmap(nf, N, pns[rd], QPs_p[rd]);
    km_p[rd] = generate_kmap(nf, N, pns[rd], QPs_p[rd]);
  }
}

void MinimalPlan::gen_inner_plan(int32_t flags) {
  for (int64_t r = region_start; r <= region_end; r++) {
    int64_t nt = n[r];
    plan_1d(nt, r, flags);
  }
}

void MinimalPlan::execute_multid_plan(MinAlignedVector& Y, MinAlignedVector& X,
                                      int32_t region_start, int32_t region_end, int64_t bp,
                                      int64_t stride) const {
  MFFTELEM* Y_data = Y.data();
  MFFTELEM* X_data = X.data();

  MinAlignedVector copy_X(X);

  MDArray YMD = create_mdarray(Y_data, n, n_dims);
  MDArray XMD = create_mdarray(X_data, n, n_dims);

  int i;
  for (i = 0; i <= region_end - region_start; ++i) {
    if ((i & 1) == 0)
      do_fft_planned(*this, &YMD, &XMD, region_start + i);
    else
      do_fft_planned(*this, &XMD, &YMD, region_start + i);
  }

  if ((i & 1) == 0) {
    // last result is in XMD, so swap XMD and YMD
    MFFTELEM* tmp = XMD.data;
    XMD.data = YMD.data;
    YMD.data = tmp;
  }

  if (YMD.data != Y.data()) {
    swap(Y, X);  // swap input Y and X, so that result is in Y
  }

  if (flags & P_INPLACE)
    X = Y;
  else if (X != copy_X) {
    X = copy_X;
  }
}

void MinimalPlan::execute_plan_no_copy(MFFTELEM** YY, MFFTELEM** XX, int64_t r, int64_t bp,
                                       int64_t stride) const {
  const int64_t* b_p = pbase[r];
  const int64_t* n_p = pns[r];
  const fft_func_t* f_p = pfunc[r];
  const int32_t* e_p = pexp[r];
  const int64_t* QPs = QPs_p[r];
  const MAP_CACHE_T* nm = nm_p[r];
  const MAP_CACHE_T* km = km_p[r];

  char nf = num_factors[r];
  minassert(nf <= MAX_FACTORS, "Too many factors to execute_plan_no_copy.");
  if (!nf) return;

  switch (nf) {
    case 1:
      f_p[0](YY, XX, n_p[0], e_p[0], bp, stride, flags);
      break;
    case 2:
      prime_factor<2>(YY, XX, N, n_p, e_p, bp, stride, flags, f_p, QPs, nm, km);
      break;
    case 3:
      prime_factor<3>(YY, XX, N, n_p, e_p, bp, stride, flags, f_p, QPs, nm, km);
      break;
    case 4:
      prime_factor<4>(YY, XX, N, n_p, e_p, bp, stride, flags, f_p, QPs, nm, km);
      break;
    case 5:
      prime_factor<5>(YY, XX, N, n_p, e_p, bp, stride, flags, f_p, QPs, nm, km);
      break;
    case 6:
      prime_factor<6>(YY, XX, N, n_p, e_p, bp, stride, flags, f_p, QPs, nm, km);
      break;
    case 7:
      prime_factor<7>(YY, XX, N, n_p, e_p, bp, stride, flags, f_p, QPs, nm, km);
      break;
    default:
      minassert(0, "Too many factors, should have planned Bluestein.");
  }
  // *YY and *XX may have flipped
}

// Execute plan function with input copying if needed
void MinimalPlan::execute_plan(MinAlignedVector& Y, MinAlignedVector& X, int32_t r, int64_t bp,
                               int64_t stride) const {
  MFFTELEM* Y_data = Y.data();
  MFFTELEM* X_data = X.data();

  MFFTELEM** YY = &Y_data;
  MFFTELEM** XX = &X_data;
  MinAlignedVector copy_X(X);

  execute_plan_no_copy(YY, XX, r, bp, stride);

  if (*YY != Y.data()) {
    swap(Y, X);
  }

  if (flags & P_INPLACE)
    X = Y;
  else if (X != copy_X)
    X = copy_X;
}
