// indexer.c - column-major indexing

#include "CPPMinimalFFT.hpp"
#include "plan.hpp"
#include <stdlib.h>
#include <string.h>
#include <type_traits>

void compute_strides(int64_t *strides, const int64_t *dims, int64_t ndims,
                     int64_t instride) {
  int64_t prod = 1;
  for (int64_t s = 0; s < ndims; s++) {
    strides[s] = instride * prod;
    prod *= dims[s];
  }
}

int64_t compute_product(const int64_t *dims, int64_t ndims) {
  int64_t sz = 1;
  for (int64_t d = 0; d < ndims; d++) {
    sz *= dims[d];
  }
  return sz;
}

int64_t indexer_count(int64_t r, int64_t ndims, int64_t *counts,
                      int64_t *strides, int64_t bp, const int64_t *dims) {
  int64_t i = 0;
  while (i < ndims) {
    if (i != r) {
      counts[i] += 1;
      if (counts[i] == dims[i]) {
        counts[i] = 0;
        bp -= strides[i] * (dims[i] - 1);
      } else {
        bp += strides[i];
        break;
      }
    }
    i++;
  }
  return (i == ndims) ? -1 : bp;
}

void do_1d_plan(MinimalPlan &P, MDArray *oy, MDArray *ix, int64_t r) {
  int64_t ndims = oy->ndims;
  int64_t *strides = (int64_t *)malloc((uint64_t)ndims * sizeof(int64_t));
  int64_t *counts = (int64_t *)calloc((uint64_t)ndims, sizeof(int64_t));

  compute_strides(strides, oy->dims, ndims, 1);

  int64_t bp = 0;
  bool flipped = false;

  MFFTELEM **YY = &(oy->data);
  MFFTELEM **XX = &(ix->data);
  MFFTELEM *orig_y = *YY;

  int64_t stride = strides[r];

  while (bp != -1) {
    P.execute_plan(YY, XX, r, bp, stride);

    if (*YY != orig_y) {
      flipped = true;
      swap_ptrs(YY, XX);
    }
    bp = indexer_count(r, ndims, counts, strides, bp, oy->dims);
  }

  if (flipped) {
    swap_ptrs(YY, XX);
  }

  free(strides);
  free(counts);
}

void do_1d_r0(MinimalPlan &P, MDArray *oy, MDArray *ix) {
  int64_t vlength = oy->dims[0];
  int64_t bp = 0;
  int64_t limit = oy->total_size;
  bool flipped = false;

  MFFTELEM **YY = &(oy->data);
  MFFTELEM **XX = &(ix->data);
  MFFTELEM *orig_y = *YY;

  while (bp < limit) {
    P.execute_plan(YY, XX, 0, bp, 1);

    if (*YY != orig_y) {
      flipped = true;
      swap_ptrs(YY, XX);
    }
    bp += vlength;
  }

  if (flipped) {
    swap_ptrs(YY, XX);
  }
}

// do_fft_planned function
void do_fft_planned(MinimalPlan &P, MDArray *oy, MDArray *ix, int64_t r) {
  if (r == 0) {
    do_1d_r0(P, oy, ix);
  } else {
    do_1d_plan(P, oy, ix, r);
  }
}

// do_1d function without plan
template <typename Func>
void do_1d_func(MDArray *oy, MDArray *ix, const fft_func_t *fs,
                const int64_t *Ns, int64_t ndims, const int32_t *es, int64_t r, int64_t bp,
                int64_t instride, int32_t flags) {
  int64_t oy_ndims = oy->ndims;
  int64_t *oy_dims = oy->dims;
  int64_t *strides = (int64_t *)malloc((uint64_t)oy_ndims * sizeof(int64_t));
  int64_t *counts = (int64_t *)calloc((uint64_t)oy_ndims, sizeof(int64_t));

  compute_strides(strides, oy_dims, oy_ndims, instride);

  int64_t vlength = oy->dims[r];
  bool flipped = false;

  MFFTELEM **YY = &(oy->data);
  MFFTELEM **XX = &(ix->data);
  MFFTELEM *orig_y = *YY;

  int64_t stride = strides[r];

  while (bp != -1) {
    if constexpr (std::is_same_v<Func, fft_func_t>) {
      fs[r](YY, XX, vlength, es[r], bp, stride, flags);
    } else if constexpr (std::is_same_v<Func, pfa2_t>) {
      prime_factor_2(YY, XX, es, Ns, fs, bp, stride, flags);
    } else if constexpr (std::is_same_v<Func, pfa3_t>) {
      prime_factor_3(YY, XX, es, Ns, fs, bp, stride, flags);
    } else {
      static_assert(0, "Unsupported function type");
    }

    if (*YY != orig_y) {
      flipped = true;
      swap_ptrs(YY, XX);
    }

    bp = indexer_count(r, oy_ndims, counts, strides, bp, oy_dims);
  }

  if (flipped) {
    swap_ptrs(YY, XX);
  }

  free(strides);
  free(counts);
}

template <typename Func>
void do_1d_r0_func(MDArray *oy, MDArray *ix, const fft_func_t *fs,
                   const int64_t *Ns, int64_t ndims, const int32_t *es, int64_t bp,
                   int64_t stride, int32_t flags) {
  int64_t vlength = oy->dims[0];
  int64_t flipped = 0;

  MFFTELEM **YY = &(oy->data);
  MFFTELEM **XX = &(ix->data);
  MFFTELEM *orig_y = *YY;

  int64_t limit = bp + oy->total_size * stride;

  while (bp < limit) {
    if constexpr (std::is_same_v<Func, fft_func_t>) {
      fs[0](YY, XX, vlength, es[0], bp, stride, flags);
    } else if constexpr (std::is_same_v<Func, pfa2_t>) {
      prime_factor_2(YY, XX, es, Ns, fs, bp, stride, flags);
    } else if constexpr (std::is_same_v<Func, pfa3_t>) {
      prime_factor_3(YY, XX, es, Ns, fs, bp, stride, flags);
    } else {
      static_assert(0, "Unsupported function type");
    }

    if (*YY != orig_y) {
      flipped = 1;
      swap_ptrs(YY, XX);
    }
    bp += stride * vlength;
  }

  if (flipped) {
    swap_ptrs(YY, XX);
  }
}

// do_fft function
template <typename Func>
void do_fft(MDArray *oy, MDArray *ix, const fft_func_t *fs, const int64_t *Ns,
            int64_t ndims, const int32_t *es, int64_t r, int64_t bp, int64_t stride,
            int32_t flags) {
  if (r == 0) {
    do_1d_r0_func<Func>(oy, ix, fs, Ns, ndims, es, bp, stride, flags);
  } else {
    do_1d_func<Func>(oy, ix, fs, Ns, ndims, es, r, bp, stride, flags);
  }
}

// explicit template instantiations

template void do_fft<fft_func_t>(MDArray *oy, MDArray *ix,
                                 const fft_func_t *fs, const int64_t *Ns, int64_t ndims,
                                 const int32_t *es, int64_t r, int64_t bp,
                                 int64_t stride, int32_t flags);

template void do_fft<pfa2_t>(MDArray *oy, MDArray *ix,
                                      const fft_func_t *fs, const int64_t *Ns,
                                      int64_t ndims, const int32_t *es, int64_t r,
                                      int64_t bp, int64_t stride,
                                      int32_t flags);

template void do_fft<pfa3_t>(MDArray *oy, MDArray *ix,
                                      const fft_func_t *fs, const int64_t *Ns,
                                      int64_t ndims, const int32_t *es, int64_t r,
                                      int64_t bp, int64_t stride,
                                      int32_t flags);
