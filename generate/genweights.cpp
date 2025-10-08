
#include <mpfr.h>

#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "../src/CPPMinimalFFT.hpp"
#include "../src/plan.hpp"

static const int FP_BITS = 256;
static const int EXP_LIMIT = 40;

// generate weights.h file with DFT coefficients

std::ostream &operator<<(std::ostream &os, const mpfr_t &val) {
  // Get buffer size needed
  size_t n = mpfr_snprintf(nullptr, 0, "%.16RNe", val);

  // Allocate buffer and format
  std::vector<char> buffer(n + 1);
  mpfr_snprintf(buffer.data(), n + 1, "%.16RNe", val);

  // Output to stream
  os << buffer.data();
  return os;
}

void print_value(const int N, int wrap, int &i, const mpfr_t &c,
                 const mpfr_t &s) {
  std::cout << c << "," << s;
  if (i != (N - 1)) {
    std::cout << ",";
    if ((++i % wrap) == 0) {
      std::cout << "\n";
    }
  }
}

void print_value(const int N, int wrap, int &i, const double &c,
                 const double &s) {
  std::cout << c << "," << s;
  if (i != (N - 1)) {
    std::cout << ",";
    if ((++i % wrap) == 0) {
      std::cout << "\n";
    }
  }
}

void normalize(mpfr_t &v) {
  mpfr_t r;
  mpfr_init2(r, FP_BITS);
  mpfr_set_d(r, 1e-50, MPFR_RNDN);
  if (mpfr_cmpabs(v, r) < 0) {
    if (mpfr_signbit(v))
      mpfr_set_d(v, -0.0, MPFR_RNDN);
    else
      mpfr_set_d(v, 0.0, MPFR_RNDN);
  }
  mpfr_clear(r);
}

void gen_weights(std::string header, int64_t N, int explimit) {
  std::cout << header;
  if (N == 0) {
    std::cout << "};\n";
    return;
  }

  mpfr_t u, d, arg, s, c;
  mpfr_init2(u, FP_BITS);
  mpfr_init2(d, FP_BITS);
  mpfr_init2(arg, FP_BITS);
  mpfr_init2(s, FP_BITS);
  mpfr_init2(c, FP_BITS);

  // u = -2.0 * M_PI / N
  mpfr_const_pi(u, MPFR_RNDN);
  mpfr_mul_ui(u, u, 2, MPFR_RNDN);
  mpfr_neg(u, u, MPFR_RNDN);
  mpfr_div_ui(u, u, N, MPFR_RNDN);

  mpfr_set_d(d, 1.0, MPFR_RNDN);

  int i = 0;

  mpfr_set_d(s, 0.0, MPFR_RNDN);
  mpfr_set_d(c, 1.0, MPFR_RNDN);

  for (int v = 0; v < explimit; ++v) {
    mpfr_mul_ui(arg, u, v, MPFR_RNDD);
    mpfr_sin(s, arg, MPFR_RNDN);
    mpfr_cos(c, arg, MPFR_RNDN);
    normalize(s);
    normalize(c);
    print_value(explimit, 1, i, c, s);
  }
  std::cout << "};\n";

  mpfr_clear(arg);
  mpfr_clear(s);
  mpfr_clear(c);

  mpfr_clear(u);
  mpfr_clear(d);
}

void gen_cossin(std::string header, int64_t radix, int explimit) {
  std::cout << header;
  if (radix == 0) {
    std::cout << "};\n";
    return;
  }

  mpfr_t u, d, arg, s, c;
  mpfr_init2(u, FP_BITS);
  mpfr_init2(d, FP_BITS);
  mpfr_init2(arg, FP_BITS);
  mpfr_init2(s, FP_BITS);
  mpfr_init2(c, FP_BITS);

  // u = -2.0 * M_PI
  mpfr_const_pi(u, MPFR_RNDN);
  mpfr_mul_ui(u, u, 2, MPFR_RNDN);
  mpfr_neg(u, u, MPFR_RNDN);

  mpfr_set_d(d, 1.0, MPFR_RNDN);
  mpfr_mul_ui(d, d, radix, MPFR_RNDU);
  mpfr_div(u, u, d, MPFR_RNDN);  // U = -2.0 * M_PI / radix
  mpfr_set_d(d, 1.0, MPFR_RNDN);

  int i = 0;

  mpfr_set_d(s, 0.0, MPFR_RNDN);
  mpfr_set_d(c, 1.0, MPFR_RNDN);

  for (int v = 0; v < explimit; ++v) {
    print_value(explimit, 1, i, c, s);
    mpfr_mul_ui(d, d, radix, MPFR_RNDU);
    mpfr_div(arg, u, d, MPFR_RNDD);
    mpfr_sin(s, arg, MPFR_RNDN);
    mpfr_cos(c, arg, MPFR_RNDN);
    normalize(s);
    normalize(c);
  }
  std::cout << "};\n";

  mpfr_clear(arg);
  mpfr_clear(s);
  mpfr_clear(c);

  mpfr_clear(u);
  mpfr_clear(d);
}

void generate_direct_entries(std::vector<std::string> &names, int N) {
  names.push_back("DIRECT_COEF_" + std::to_string(N));

  int i = 0;

  gen_weights("alignas(ALIGN_SZ) static const MFFTELEMRI DIRECT_COEF_" +
                  std::to_string(N) + "[] = {\n",
              N, N);
  std::cout << "\n";
}

void generate_direct() {
  std::cout << "\n#define ALIGN_SZ 16\n\n";
  std::vector<std::string> names;
  for (int n = 0; n <= DIRECT_SZ; n++) {
    generate_direct_entries(names, n);
  }
  std::cout << "static const MFFTELEMRI *DIRECT_COEFFS[] = {\n";
  for (int i = 0; i < names.size(); ++i) {
    std::cout << names[i];
    if (i != names.size() - 1) {
      std::cout << ",";
      if ((i + 1) % 4 == 0) {
        std::cout << "\n";
      }
    }
  }
  std::cout << "};\n\n";
}

int main() {
  constexpr auto max_precision{std::numeric_limits<long double>::digits10 + 1};

  std::cout << std::scientific << std::setprecision(max_precision);
  std::cout << "\n#include \"CPPMinimalFFT.hpp\"\n";
  generate_direct();

  std::cout << "static const int COSSIN_EXP_LIMIT = " << EXP_LIMIT << ";\n\n";

  std::string preamble = "\nalignas (ALIGN_SZ) static const double COS_SIN_";
  gen_cossin(preamble + "2[] = {\n", 2, EXP_LIMIT);
  gen_cossin(preamble + "3[] = {\n", 3, EXP_LIMIT);
  gen_cossin(preamble + "5[] = {\n", 5, EXP_LIMIT);
  gen_cossin(preamble + "7[] = {\n", 7, EXP_LIMIT);

  std::cout << "\n";

  return 0;
}