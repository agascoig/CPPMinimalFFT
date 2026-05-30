
#ifndef __PFA_HPP__
#define __PFA_HPP__

#include <cstdint>

#include "CPPMinimalFFT.hpp"

static const int MAX_FACTORS = 7;
static const int MAX_MAP_CACHE = 1 << 14;

using MAP_CACHE_T = uint16_t;

void prime_factor(int nf, MFFTELEM** YY, MFFTELEM** XX, const int64_t N, const int64_t* Ns, const int32_t* es,
                  const int64_t bp, const int64_t stride, const int32_t flags, const fft_func_t* fs,
                  const int64_t *QPs, const MAP_CACHE_T *nm, const MAP_CACHE_T *km);

int64_t* generate_QPs(const int32_t nf, const int64_t* Ns);
MAP_CACHE_T* generate_nmap(const int nf, const int64_t N, const int64_t* Ns, const int64_t *QPs);
MAP_CACHE_T* generate_kmap(const int nf, const int64_t N, const int64_t* Ns, const int64_t *QPs);

static void print_fns(char* buf, fft_func_t* fns) {
  char fn_str[256];
  buf[0] = '\0';
  fn_str[0] = '\0';
  if (fns == nullptr) return;
  for (int i = 0; i < MAX_FACTORS; ++i) {
    fft_func_t func = fns[i];
    if (func == nullptr) continue;
    for (int j = 0; j < sizeof(fns_names) / sizeof(fn_name_s); ++j) {
      if (func == fns_names[j].fn) {
        snprintf(fn_str, sizeof(fn_str), "%s ", fns_names[j].name);
        strcat(buf, fn_str);
        break;
      }
    }
  }
  if (strlen(buf)) buf[strlen(buf) - 1] = 0;  // remove last space
}

#endif