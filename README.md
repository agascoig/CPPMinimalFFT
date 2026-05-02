
# CPPMinimalFFT

This is a simple FFT library written in C++, for the study of
the performance compared to MinimalFFT.jl to make improvements.

Google Highway is optionally used for SIMD operations for single
precision by 4, or double precision by 2.

*sincos* is borrowed from the Openlibm library for
better performance than the standard C library.

FFTW is a dependency for the correctness and performance testing in the test
subdirectory.

This was originally written in C and converted to C++.

Compiling with gcc or fastmath yields much better performance
than standard clang.

## Organization

| Function | |
|---------------------|-------------------------------------------|
| Lowest level functions | direct_dft, bluestein, fftr2, fftr3, etc. |
| Mid level decomposition functions | prime_factor_2, prime_factor_3 |
| Indexer functions | do_1d, do_1d_r0 |
| Multi-dimensional FFT indexers | do_fft_planned, do_1d |
| Planning functions | create_min_plan, execute_plan |

## Testing

The only tests currently run are test/test17.cpp.

(No testing yet for multi-dimensional FFTs.)

## License

The license is MIT as described in LICENSE.txt, except for
sincos.cpp which requires some attributions.
