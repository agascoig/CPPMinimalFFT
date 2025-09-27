
# CMinimalFFT

This is a simple FFT library written in C++, currently
supporting only the complex double type.  The purpose of
this repository is to study the performance compared to
MinimalFFT.jl.  sincos is from the Openlibm library for
better performance.  Compiling with gcc also yields
much better performance than clang.

## Organization

| Function | |
|---------------------|-------------------------------------------|
| Lowest level functions | direct_dft, bluestein, fftr2, fftr3, etc. |
| Mid level decomposition functions | prime_factor_2, prime_factor_3 |
| Indexer functions | do_1d, do_1d_r0 |
| Multi-dimensional FFT indexers | do_fft_planned, do_1d |
| Planning functions | create_min_plan, execute_plan |

## Testing

The only tests currently run are test/test17.cpp.  No testing yet for multi-dimensional FFTs.

## License

The license is MIT as described in LICENSE.txt.
