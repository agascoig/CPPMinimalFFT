
#ifndef __PFA_HPP__
#define __PFA_HPP__

#include <cstdint>

#include "CPPMinimalFFT.hpp"

template <int nf>
void prime_factor(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
                  const int64_t stride, const int32_t flags) noexcept;

void prime_factor(MFFTELEM** YY, MFFTELEM** XX, const region_data& rd, const int64_t bp,
                  const int64_t stride, const int32_t flags, const int32_t nf) noexcept;

void QPs(int64_t* params, const int32_t nf, const int64_t* Ns) noexcept;
MAP_CACHE_T* generate_nmap(const int nf, const int64_t N, const int64_t* Ns,
                           const int64_t* QPs) noexcept;
MAP_CACHE_T* generate_kmap(const int nf, const int64_t N, const int64_t* Ns,
                           const int64_t* QPs) noexcept;

#endif