
#pragma once

// Source: chatgpt

typedef struct {
  long long n;       // count of values
  double sum;        // running sum
  double sum_recip;  // running sum of reciprocals
  double sum_logs;   // for geometric mean
} HarmonicMean;

void hm_init(HarmonicMean *hm);

void hm_add(HarmonicMean *hm, double x);

double hm_value(const HarmonicMean *hm);

double gm_value(const HarmonicMean *hm);

double m_value(const HarmonicMean *hm);
