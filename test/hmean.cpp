
#include "hmean.hpp"

#include <stdio.h>

#include <cmath>

// Source: chatgpt

// initialize
void hm_init(HarmonicMean *hm) {
  hm->n = 0;
  hm->sum_recip = 0.0;
}

// add a new value (must be > 0)
void hm_add(HarmonicMean *hm, double x) {
  if (x <= 0.0) {
    fprintf(stderr,
            "Error: harmonic mean undefined for non-positive values.\n");
    return;
  }
  hm->n += 1;
  hm->sum += x;
  hm->sum_recip += 1.0 / x;
  hm->sum_logs += log(x);
}

// get current harmonic mean
double hm_value(const HarmonicMean *hm) {
  if (hm->n == 0 || hm->sum_recip == 0.0) return 0.0;  // no data yet
  return (double)hm->n / hm->sum_recip;
}

double gm_value(const HarmonicMean *hm) {
  if (hm->n == 0) return 0.0;  // no data yet
  return exp(hm->sum_logs / hm->n);
}

double m_value(const HarmonicMean *hm) {
  if (hm->n == 0) return 0.0;  // no data yet
  return hm->sum / hm->n;
}