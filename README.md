
# CPPMinimalFFT

This is a simple FFT library written in C++ intended to be used for
performance studies.  The approach used is the
prime-factor algorithm, with Spiral [1] to generate the FFT routines,
which are embedded in Stockham auto-sorting routines.

Google Highway for SIMD (currently only 128-bit supported and
tested on ARM NEON).

This library was originally written in C and converted to C++.

## Organization

| Function | |
|---------------------|-------------------------------------------|
| Lowest level functions | direct_dft, bluestein, fftr2, fftr3, etc. |
| Mid level prime-factor algorithm | prime_factor |
| Indexer functions | do_fft_planned, do_fft |
| Planning functions (preferred interface) | MinimalPlan class |

## Testing

| Test | Description     |
|------|-----------------|
| test1 | Test spiral, direct DFT, and small DFT up to N=31 to see if DFT matrix can be reconstructed. |
| test2 | Sweep test for N=1..50653. |
| test3 | Performance test N = 3780 for profiling. |
| test4 | Test N = 3780 comparing with FFTW. |
| test17 | Mixed testbench including small, direct, Stockham, prime-factor algorithm (PFA), Bluestein, and planned tests. Optionally benchmarked. |
| test18 | Multi-dimensional tests. |
| test19 | Partial multi-dimensional tests. |

FFTW is a dependency for the correctness and performance
testing in the test subdirectory.

test2 with single-precision, SIMD currently fails due to
Bluestein numerical issues. Therefore, s4/bluestein.cpp
is not in the build.

## License

The license is MIT as described in LICENSE.txt.

## Performance Considerations for Version v0.1.0

* An interleaved, complex form is used for the low-level FFT routines, which is
  reportedly slower than separate real and imaginary calculations.
* The weights are generally calculated when needed, and not
  pre-calculated in tables or cached.
* The SIMD approach used is simply to unroll loops, and is not
  highly optimized.
* Profiling shows almost all of the time is consumed in the
  low-level FFT routines.  The Stockham approach uses
  2N memory compared to the Cooley-Tukey algorithm (less cache friendly).
* For version V0.1.0, the performance is roughly half that of FFTW.
  
## Spiral

The Spiral FFT generator is used to generate complex FFT butterflies.
Up to twenty variants are tried for each radix and benchmarked by a separate program
(Spiral does not support Apple Silicon), according to the following
template:

```
    opts := CplxSpiralDefaults;
    opts.realVect := false;
    opts.useDeref := false;
    RandomSeed(0);
    transform := DFT(4, -1);
    ruletree := RandomRuleTree(transform, opts);
    icode := CodeRuleTree(ruletree, opts);
    PrintCode("fftr4", icode, opts);
```

The DFT(N, -1), negative one, indicates the forward FFT.

Custom scripts are used to translate the C generated code into scalar
and SIMD code.

## References

[1] Spiral FFT Generator https://www.spiral.net

[2] Google Highway SIMD https://github.com/google/highway

[3] C. -F. Hsiao, Y. Chen and C. -Y. Lee, "A Generalized Mixed-Radix Algorithm for Memory-Based FFT Processors," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 57, no. 1, pp. 26-30, Jan. 2010, doi: 10.1109/TCSII.2009.2037262 https://ieeexplore.ieee.org/document/5373949

[4] Chirp Z-transform (Bluestein algorithm) https://en.wikipedia.org/wiki/Chirp_Z-transform