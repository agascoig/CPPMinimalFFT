#include "CPPMinimalFFT.hpp"
#include "plan.hpp"
#include <complex>
#include <hwy/highway.h>
#include <stdint.h>

// slows down on m4 pro: simd 0.348e-7us  N = 15.  vs. 0.298e-7us for scalar
namespace hn = hwy::HWY_NAMESPACE;

typedef hn::CappedTag<double, 2> CappedTagDouble2;

typedef struct {
    std::complex<double> B;
    int64_t N;
    int32_t inverse;
} direct_buffer_t;

// Global buffer for caching twiddle factor
static direct_buffer_t direct_buff = {std::complex<double>(0.0, 0.0), 0, 0};

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {

// Multiply two interleaved complex<double> (2 doubles per complex)
static inline hn::Vec<hn::CappedTag<double, 2>> 
ComplexMul(hn::CappedTag<double, 2> d,
           hn::Vec<hn::CappedTag<double, 2>> a,
           hn::Vec<hn::CappedTag<double, 2>> b) {
  // Split into real/imag parts
  auto ar = hn::DupEven(a);  // [ar, ar]
  auto ai = hn::DupOdd(a);   // [ai, ai]
  auto br = hn::DupEven(b);  // [br, br]
  auto bi = hn::DupOdd(b);   // [bi, bi]

  // Complex multiply: (ar*br - ai*bi, ar*bi + ai*br)
//  auto real = hn::Sub(hn::Mul(ar, br), hn::Mul(ai, bi));
  auto real = hn::NegMulAdd(ai, bi, hn::Mul(ar, br));
  auto imag = hn::Add(hn::Mul(ar, bi), hn::Mul(ai, br));

  // Re-interleave [real, imag]
  return hn::OddEven(imag, real);
}

// Direct DFT implementation
void direct_dft(MFFTELEM **Y, MFFTELEM **X, int64_t N, int32_t e1, int64_t bp, int64_t stride, int32_t inverse) {
    MFFTELEM *__restrict__ y = *Y;
    MFFTELEM *__restrict__ x = *X;

    using D = hn::CappedTag<double, 2>;
    D d;

    alignas(16) double one[2] = {1.0, 0.0};
    alignas(16) double zero[2] = {0.0, 0.0};
    std::complex<double> B;
    
    // Check if we can reuse cached twiddle factor
    if (direct_buff.N == N) {
        B = direct_buff.B;
        if (inverse != direct_buff.inverse) {
            B = std::conj(B);
            direct_buff.B = B;
            direct_buff.inverse = inverse;
        }
    } else {
        // Compute new twiddle factor
        B = minsincos(inverse ? 2.0 * M_PI / N : -2.0 * M_PI / N);
        
        direct_buff.B = B;
        direct_buff.N = N;
        direct_buff.inverse = inverse;
    }
    
    auto W_step = hn::Load(d, one);
    auto B_vec = hn::Load(d, reinterpret_cast<const double*>(&B));

    for (int64_t k = 0; k < N; k++) {
        auto W = hn::Load(d, one);
        auto s = hn::Load(d, zero);

        for (int64_t n = 0; n < N; n++) {
            auto xval = hn::Load(d, reinterpret_cast<const double*>(&x[bp + stride * n]));
            auto t = ComplexMul(d, W, xval);
            s = hn::Add(s, t);
            W = ComplexMul(d, W, W_step);
        }   
        hn::Store(s, d, reinterpret_cast<double*>(&y[bp + stride * k]));
        W_step = ComplexMul(d, W_step, B_vec);
    }    
}

}
  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

void direct_dft(MFFTELEM **Y, MFFTELEM **X, int64_t N, int32_t e1, int64_t bp, int64_t stride, int32_t inverse) {
    return N_NEON_WITHOUT_AES::direct_dft(Y, X, N, e1, bp, stride, inverse);
}
