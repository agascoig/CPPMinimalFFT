// indexer.c - column-major indexing

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"

static inline int64_t indexer_count(const int32_t r, const int32_t ndims,
                                    int64_t *__restrict__ counts,
                                    const int64_t *__restrict__ strides,
                                    int64_t bp,
                                    const int64_t *__restrict__ dims) {
  int32_t i = 0;

  while (i < ndims) {
    if (i != r) {
      *counts += 1;
      bp += *strides;
      if (*counts != *dims)
        return bp;
      else {
        *counts = 0;
        bp -= *strides * (*dims);
      }
    }
    i++;
    counts++;
    dims++;
    strides++;
  }
  return -1;
}

void do_1d_plan(const MinimalPlan &P, MDArray *oy, MDArray *ix, int32_t r) {
  int64_t strides[MAX_DIMS];
  int64_t counts[MAX_DIMS] = {0};

  const int32_t ndims = oy->ndims;
  int64_t prod = 1;
  int64_t *dims_p = oy->dims;
  for (int32_t s = 0; s < ndims; s++) {
    strides[s] = prod;
    prod *= *dims_p++;
  }

  MFFTELEM **YY = &(oy->data);
  MFFTELEM **XX = &(ix->data);
  MFFTELEM *orig_y = *YY;

  const int64_t stride = strides[r];
  int64_t bp = 0;

  while (bp != -1) {
    P.execute_plan_no_copy(YY, XX, r, bp, stride);

    if (*YY != orig_y) {
      orig_y = *YY;
      *YY = *XX;
      *XX = orig_y;
      orig_y = nullptr;  // mark flipped
    }
    bp = indexer_count(r, ndims, counts, strides, bp, dims_p);
  }

  if (orig_y == nullptr) {
    orig_y = *YY;
    *YY = *XX;
    *XX = orig_y;
  }
}

void do_1d_r0(const MinimalPlan &P, MDArray *oy, MDArray *ix) {
  const int64_t vlength = oy->dims[0];
  const int64_t limit = oy->total_size;

  MFFTELEM **YY = &(oy->data);
  MFFTELEM **XX = &(ix->data);
  MFFTELEM *orig_y = *YY;

  int64_t bp = 0;

  while (bp < limit) {
    P.execute_plan_no_copy(YY, XX, 0, bp, 1);

    if (*YY != orig_y) {
      orig_y = *YY;
      *YY = *XX;
      *XX = orig_y;
      orig_y = nullptr;  // mark flipped
    }
    bp += vlength;
  }

  if (orig_y == nullptr) {
    orig_y = *YY;
    *YY = *XX;
    *XX = orig_y;
  }
}

// do_fft_planned function
void do_fft_planned(const MinimalPlan &P, MDArray *oy, MDArray *ix, int32_t r) {
  if (r == 0)
    do_1d_r0(P, oy, ix);
  else
    do_1d_plan(P, oy, ix, r);
}

// do_1d function without plan
template <typename Func>
void do_1d_func(MDArray *oy, MDArray *ix, const int64_t *Ns, const int32_t *es,
                int64_t bp, int64_t instride, int32_t flags,
                const fft_func_t *fs, const int64_t *params, int32_t r) {
  int64_t *oy_dims = oy->dims;
  int64_t strides[MAX_DIMS];
  int64_t counts[MAX_DIMS] = {0};

  const int64_t oy_ndims = oy->ndims;
  int64_t prod = 1;
  int64_t *dims_p = oy->dims;
  for (int32_t s = 0; s < oy_ndims; s++) {
    strides[s] = instride * prod;
    prod *= *dims_p++;
  }

  MFFTELEM **YY = &(oy->data);
  MFFTELEM **XX = &(ix->data);
  MFFTELEM *orig_y = *YY;

  const int64_t stride = strides[r];
  const fft_func_t fsr = fs[r];
  const int32_t esr = es[r];
  const int64_t vlength = oy_dims[r];

  while (bp != -1) {
    if constexpr (std::is_same_v<Func, fft_func_t>) {
      fsr(YY, XX, vlength, esr, bp, stride, flags);
    } else if constexpr (std::is_same_v<Func, pfa2_t>) {
      prime_factor_2(YY, XX, Ns, es, bp, stride, flags, fs, params);
    } else if constexpr (std::is_same_v<Func, pfa3_t>) {
      prime_factor_3(YY, XX, Ns, es, bp, stride, flags, fs, params);
    } else {
      static_assert(0, "Unsupported function type");
    }

    if (*YY != orig_y) {
      orig_y = *YY;
      *YY = *XX;
      *XX = orig_y;
      orig_y = nullptr;  // mark flipped
    }

    bp = indexer_count(r, oy_ndims, counts, strides, bp, oy_dims);
  }

  if (orig_y == nullptr) {
    orig_y = *YY;
    *YY = *XX;
    *XX = orig_y;
  }
}

template <typename Func>
void do_1d_r0_func(MDArray *oy, MDArray *ix, const int64_t *Ns,
                   const int32_t *es, int64_t bp, int64_t stride, int32_t flags,
                   const fft_func_t *fs, const int64_t *params) {
  MFFTELEM **YY = &(oy->data);
  MFFTELEM **XX = &(ix->data);
  MFFTELEM *orig_y = *YY;

  const int64_t limit = bp + oy->total_size * stride;

  const fft_func_t fs0 = fs[0];
  const int32_t es0 = es[0];
  const int64_t vlength = oy->dims[0];

  while (bp < limit) {
    if constexpr (std::is_same_v<Func, fft_func_t>) {
      fs0(YY, XX, vlength, es0, bp, stride, flags);
    } else if constexpr (std::is_same_v<Func, pfa2_t>) {
      prime_factor_2(YY, XX, Ns, es, bp, stride, flags, fs, params);
    } else if constexpr (std::is_same_v<Func, pfa3_t>) {
      prime_factor_3(YY, XX, Ns, es, bp, stride, flags, fs, params);
    } else {
      static_assert(0, "Unsupported function type");
    }

    if (*YY != orig_y) {
      orig_y = *YY;
      *YY = *XX;
      *XX = orig_y;
      orig_y = nullptr;  // mark flipped
    }
    bp += stride * vlength;
  }

  if (orig_y == nullptr) {
    orig_y = *YY;
    *YY = *XX;
    *XX = orig_y;
  }
}

// do_fft function
template <typename Func>
void do_fft(MDArray *oy, MDArray *ix, const int64_t *Ns, const int32_t *es,
            int64_t bp, const int64_t stride, const int32_t flags,
            const fft_func_t *fs, const int64_t *params, const int32_t r) {
  if (r == 0)
    do_1d_r0_func<Func>(oy, ix, Ns, es, bp, stride, flags, fs, params);
  else
    do_1d_func<Func>(oy, ix, Ns, es, bp, stride, flags, fs, params, r);
}

// explicit template instantiations

template void do_fft<fft_func_t>(MDArray *oy, MDArray *ix, const int64_t *Ns,
                                 const int32_t *es, int64_t bp, int64_t stride,
                                 int32_t flags, const fft_func_t *fs,
                                 const int64_t *params, int32_t r);

template void do_fft<pfa2_t>(MDArray *oy, MDArray *ix, const int64_t *Ns,
                             const int32_t *es, int64_t bp, int64_t stride,
                             int32_t flags, const fft_func_t *fs,
                             const int64_t *params, int32_t r);

template void do_fft<pfa3_t>(MDArray *oy, MDArray *ix, const int64_t *Ns,
                             const int32_t *es, int64_t bp, int64_t stride,
                             int32_t flags, const fft_func_t *fs,
                             const int64_t *params, int32_t r);