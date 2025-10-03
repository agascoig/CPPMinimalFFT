
#pragma once

// Source: chatgpt

typedef struct {
    long long n;    // count of values
    double sum_recip; // running sum of reciprocals
} HarmonicMean;

void hm_init(HarmonicMean *hm);

void hm_add(HarmonicMean *hm, double x);

double hm_value(const HarmonicMean *hm);

