
#ifndef __PFA_HPP__
#define __PFA_HPP__

#include <cstdint>

#include "CPPMinimalFFT.hpp"

static const int MAX_FACTORS = 7;
static const int MAX_MAP_CACHE = 1 << 12;

using MAP_CACHE_T = uint16_t;

template <int nf>
void prime_factor(MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int64_t* Ns, const int32_t* es,
                  const int64_t bp, const int64_t stride, const int32_t flags, const fft_func_t* fs,
                  const int64_t *QPs, const MAP_CACHE_T *nm, const MAP_CACHE_T *km) noexcept;

int64_t* generate_QPs(const int32_t nf, const int64_t* Ns) noexcept;
MAP_CACHE_T* generate_nmap(const int nf, const int64_t N, const int64_t* Ns, const int64_t *QPs) noexcept;
MAP_CACHE_T* generate_kmap(const int nf, const int64_t N, const int64_t* Ns, const int64_t *QPs) noexcept;

#endif