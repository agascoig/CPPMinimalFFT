// indexer.c - column-major indexing

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"

static inline int64_t indexer_count(const int32_t ndims, int64_t* __restrict__ counts,
                                    const int64_t* __restrict__ strides, int64_t bp,
                                    const int64_t* __restrict__ dims) noexcept {
  int32_t i = 0;

  // r dimension removed here

  while (i < ndims) {
    *counts += 1;
    bp += *strides;
    if (*counts != *dims)
      return bp;
    else {
      *counts = 0;
      bp -= *strides * (*dims);
    }
    i++;
    counts++;
    dims++;
    strides++;
  }

  return -1;
}

void do_1d_plan(const MinimalPlan& P, MDArray* oy, MDArray* ix, int32_t r) noexcept {
  int64_t counts[MAX_DIMS] = {0};

  int32_t ndims = oy->ndims;
  int64_t* indims_p = oy->dims;
  int64_t* instrides = oy->strides;

  int64_t dims[MAX_DIMS];
  int64_t strides[MAX_DIMS];

  int64_t* dims_p = dims;
  int64_t* strides_p = strides;

  for (int d = 0; d < oy->ndims; ++d) {
    if (d != r) {
      *strides_p++ = instrides[d];
      *dims_p++ = indims_p[d];
    }
  }
  ndims--;

  MFFTELEM** YY = &(oy->data);
  MFFTELEM** XX = &(ix->data);
  MFFTELEM* orig_y = *YY;
  MFFTELEM* orig_x = *XX;

  const int64_t stride = instrides[r];
  int64_t bp = 0;

  while (bp != -1) {
    *YY = orig_y;  // keep pointing back to original data
    *XX = orig_x;  // keep pointing back to original data
    P.execute_plan_no_copy(YY, XX, r, bp, stride);
    bp = indexer_count(ndims, counts, strides, bp, dims);
  }
  // *YY and *XX may have flipped
}

void do_1d_r0(const MinimalPlan& P, MDArray* oy, MDArray* ix) noexcept {
  const int64_t vlength = oy->dims[0];
  const int64_t limit = oy->total_size;

  MFFTELEM** YY = &(oy->data);
  MFFTELEM** XX = &(ix->data);
  MFFTELEM* orig_y = *YY;
  MFFTELEM* orig_x = *XX;

  int64_t bp = 0;

  while (bp < limit) {
    *YY = orig_y;  // keep pointing back to original data
    *XX = orig_x;  // keep pointing back to original data
    P.execute_plan_no_copy(YY, XX, 0, bp, 1);
    bp += vlength;
  }
  // *YY and *XX may have flipped
}

// do_fft_planned function
void do_fft_planned(const MinimalPlan& P, MDArray* oy, MDArray* ix, int32_t r) noexcept {
  if (r == 0)
    do_1d_r0(P, oy, ix);
  else
    do_1d_plan(P, oy, ix, r);
}

// do_1d function without plan
void do_1d_func(MDArray* oy, MDArray* ix, const int64_t* Ns, const int32_t* es, int64_t bp,
                int64_t instride, int32_t flags, const fft_func_t* fs, const int64_t* params,
                int32_t r) noexcept {
  int64_t counts[MAX_DIMS] = {0};

  int32_t ndims = oy->ndims;
  const int64_t* indims_p = oy->dims;
  const int64_t* instrides = oy->strides;

  int64_t dims[MAX_DIMS];
  int64_t strides[MAX_DIMS];

  int64_t* dims_p = dims;
  int64_t* strides_p = strides;

  for (int d = 0; d < oy->ndims; ++d) {
    if (d != r) {
      *strides_p++ = instrides[d];
      *dims_p++ = indims_p[d];
    }
  }
  ndims--;

  MFFTELEM** YY = &(oy->data);
  MFFTELEM** XX = &(ix->data);
  MFFTELEM* orig_y = *YY;
  MFFTELEM* orig_x = *XX;

  const int64_t stride = instrides[r];
  const fft_func_t fsr = fs[r];
  const int32_t esr = es[r];
  const int64_t vlength = oy->dims[r];

  while (bp != -1) {
    *YY = orig_y;  // keep pointing back to original data
    *XX = orig_x;  // keep pointing back to original data
    fsr(YY, XX, vlength, esr, bp, stride, flags);
    bp = indexer_count(ndims, counts, strides, bp, dims);
  }
  // *XX and *YY may have flipped
}

void do_1d_r0_func(MDArray* oy, MDArray* ix, const int64_t* Ns, const int32_t* es, int64_t bp,
                   int64_t stride, int32_t flags, const fft_func_t* fs, const int64_t* params) noexcept {
  MFFTELEM** YY = &(oy->data);
  MFFTELEM** XX = &(ix->data);
  MFFTELEM* orig_y = *YY;
  MFFTELEM* orig_x = *XX;

  const int64_t limit = bp + oy->total_size * stride;

  const fft_func_t fs0 = fs[0];
  const int32_t es0 = es[0];
  const int64_t vlength = oy->dims[0];

  while (bp < limit) {
    *YY = orig_y;  // keep pointing back to original data
    *XX = orig_x;  // keep pointing back to original data
    fs0(YY, XX, vlength, es0, bp, stride, flags);
    bp += stride * vlength;
  }
  // *XX and *YY may have flipped
}

// do_fft function
void do_fft(MDArray* oy, MDArray* ix, const int64_t* Ns, const int32_t* es, int64_t bp,
            const int64_t stride, const int32_t flags, const fft_func_t* fs, const int64_t* params,
            const int32_t r) noexcept {
  if (r == 0)
    do_1d_r0_func(oy, ix, Ns, es, bp, stride, flags, fs, params);
  else
    do_1d_func(oy, ix, Ns, es, bp, stride, flags, fs, params, r);
}
