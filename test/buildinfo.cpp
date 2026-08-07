
#include <buildinfo.h>
#include <iostream>
#include <cxxabi.h>
#include "../src/CPPMinimalFFT.hpp"

void print_compiler_ver() {
  std::cout << "# CXX_COMPILER: " << BUILD_CXX_COMPILER << " ";
#ifdef __clang__
  std::cout << __clang_major__ << "." << __clang_minor__ << "."
            << __clang_patchlevel__;
#elif defined(__GNUC__)
  std::cout << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
  std::cout << _MSC_VER;
#else
  std::cout << "Unknown Compiler (" << __VERSION__ << ")";
#endif
  std::cout << " CXX_FLAGS: " << BUILD_CXX_FLAGS << std::endl;
  std::cout << "# BUILD_TYPE: " << BUILD_BUILD_TYPE << std::endl;
  std::cout << "# BUILD_SYSTEM: " << BUILD_SYSTEM << std::endl;
  int status = 0;
  char* demangled =
      abi::__cxa_demangle(typeid(MFFTELEM).name(), nullptr, nullptr, &status);
  std::cout << "# Vector type: " << demangled << std::endl << std::flush;
  free(demangled);
}

