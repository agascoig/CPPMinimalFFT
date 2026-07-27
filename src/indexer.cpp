// indexer.cpp - column-major indexing

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"

int64_t indexer_count(int32_t ndims, int64_t* __restrict__ counts,
                                    const int64_t* __restrict__ strides, int64_t bp,
                                    const int64_t* __restrict__ dims) noexcept {
  // ndims > 0, r dimension removed here
  do {
    (*counts)++;
    bp += *strides;
    if (*counts != *dims) [[likely]]
      return bp;
    else {
      *counts = 0;
      bp -= *strides * (*dims);
    }
    counts++;
    dims++;
    strides++;
  } while (--ndims);

  return -1;
}

void do_1d_plan(const MinimalPlan& P, MFFTELEM** YY, MFFTELEM** XX, int32_t region) noexcept {
  auto& r = P.regions[region - P.region_start];
  int64_t counts[MAX_DIMS] = {0};
  int32_t ndims = P.n_dims;
  const int64_t* indims_p = P.n;
  const int64_t* instrides = P.strides;
  int64_t dims[MAX_DIMS];
  int64_t strides[MAX_DIMS];
  int64_t* dims_p = dims;
  int64_t* strides_p = strides;
  // remove r dimension from strides, dims
  for (int d = 0; d < ndims; ++d) {
    if (d != region) {
      *strides_p++ = instrides[d];
      *dims_p++ = indims_p[d];
    }
  }
  ndims--;
  MFFTELEM* orig_y = *YY;
  MFFTELEM* orig_x = *XX;
  const int64_t stride = instrides[region];
  int64_t bp = 0;
  while (bp != -1) {
    *YY = orig_y;  // keep pointing back to original data
    *XX = orig_x;  // keep pointing back to original data
    P.execute_plan_no_copy(YY, XX, region, bp, stride);
    bp = indexer_count(ndims, counts, strides, bp, dims);
  }
  // *YY and *XX may have flipped
}

void do_1d_r0(const MinimalPlan& P, MFFTELEM** YY, MFFTELEM** XX) noexcept {
  const int64_t vlength = P.n[0];
  const int64_t limit = P.N;
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
void do_fft_planned(const MinimalPlan& P, MFFTELEM** YY, MFFTELEM** XX, int32_t region) noexcept {
  if (region == 0)
    do_1d_r0(P, YY, XX);
  else
    do_1d_plan(P, YY, XX, region);
}

template <int NDIMS>
int64_t indexer_count(int64_t* __restrict__ counts,
                      const int64_t* __restrict__ strides, int64_t bp,
                      const int64_t* __restrict__ dims) noexcept {
  // NDIMS > 0, r dimension removed here
  for (int i = 0; i < NDIMS; ++i) {
    (*counts)++;
    bp += *strides;
    if (*counts != *dims) [[likely]]
      return bp;
    else {
      *counts = 0;
      bp -= *strides * (*dims);
    }
    counts++;
    dims++;
    strides++;
  }

  return -1;
}

// do_1d function without plan
template <int NF>
void do_1d_func(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, int64_t bp,
                const int64_t instride, const int32_t flags, const int32_t r) noexcept {
  int64_t counts[NF - 1] = {0};  // MAX_FACTORS-1: r dimension removed
  const int64_t* indims_p = rd.ns;
  const int64_t* instrides = rd.strides;
  int64_t dims[NF - 1];
  int64_t strides[NF - 1];
  int64_t* dims_p = dims;
  int64_t* strides_p = strides;
  // remove r dimension from strides, dims
  for (int d = 0; d < NF; ++d) {
    if (d != r) {
      *strides_p++ = instride * instrides[d];
      *dims_p++ = indims_p[d];
    }
  }
  const int32_t ndims = NF - 1;
  MFFTELEM* orig_y = *YY;
  MFFTELEM* orig_x = *XX;
  const int64_t stride = instride * instrides[r];
  const fft_func_t fsr = rd.func[r];
  const int32_t esr = rd.exp[r];
  const int64_t vlength = rd.ns[r];
  while (bp != -1) {
    *YY = orig_y;  // keep pointing back to original data
    *XX = orig_x;  // keep pointing back to original data
    fsr(YY, XX, vlength, esr, bp, stride, flags);
    bp = indexer_count<NF - 1>(counts, strides, bp, dims);
  }
  // *XX and *YY may have flipped
}

// do_1d_r0_func without plan
template <int NF>
void do_1d_r0_func(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, int64_t bp,
                   const int64_t stride, const int32_t flags) noexcept {
  MFFTELEM* orig_y = *YY;
  MFFTELEM* orig_x = *XX;
  const int64_t limit = bp + rd.n * stride;
  const fft_func_t fs0 = rd.func[0];
  const int32_t es0 = rd.exp[0];
  const int64_t vlength = rd.ns[0];
  while (bp < limit) {
    *YY = orig_y;  // keep pointing back to original data
    *XX = orig_x;  // keep pointing back to original data
    fs0(YY, XX, vlength, es0, bp, stride, flags);
    bp += stride * vlength;
  }
  // *XX and *YY may have flipped
}

// specialization do_1d function with NF == 2
template <>
void do_1d_func<2>(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, int64_t bp,
                   const int64_t instride, const int32_t flags, const int32_t r) noexcept {
  // r == 1 here
  const int64_t stride0 = instride * rd.strides[0];
  const int64_t stride1 = instride * rd.strides[1];
  const int64_t dim0 = rd.ns[0];
  const int64_t dim1 = rd.ns[1];
  MFFTELEM* orig_y = *YY;
  MFFTELEM* orig_x = *XX;
  const int32_t esr = rd.exp[r];
  const fft_func_t fsr = rd.func[r];
  const int64_t limit = bp + instride * dim0 * stride0;
  while (bp != limit) {
    *YY = orig_y;  // keep pointing back to original data
    *XX = orig_x;  // keep pointing back to original data
    fsr(YY, XX, dim1, esr, bp, stride1, flags);
    bp += stride0;
  }
  // *XX and *YY may have flipped
}

// do_fft function without plan
template <int NF>
void do_fft(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
            const int64_t stride, const int32_t flags, const int32_t r) noexcept {
  if (r == 0)
    do_1d_r0_func<NF>(YY, XX, rd, bp, stride, flags);
  else
    do_1d_func<NF>(YY, XX, rd, bp, stride, flags, r);
}

template void do_fft<2>(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
                        const int64_t stride, const int32_t flags, const int32_t r) noexcept;

template void do_fft<3>(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
                        const int64_t stride, const int32_t flags, const int32_t r) noexcept;

template void do_fft<4>(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
                        const int64_t stride, const int32_t flags, const int32_t r) noexcept;

template void do_fft<5>(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
                        const int64_t stride, const int32_t flags, const int32_t r) noexcept;

template void do_fft<6>(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
                        const int64_t stride, const int32_t flags, const int32_t r) noexcept;

template void do_fft<7>(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
                        const int64_t stride, const int32_t flags, const int32_t r) noexcept;
