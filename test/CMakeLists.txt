# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version
project(test17 CXX)          # Specify project name and language

add_executable(test17
    test17.cpp
)

set(ASAN_FLAGS
-v
#   -fsanitize=address
#   -fno-omit-frame-pointer
)

target_compile_options(test17 PRIVATE ${ASAN_FLAGS})
target_link_options(test17 PRIVATE ${ASAN_FLAGS})

find_library(FFTW_LIB fftw3)

target_link_libraries(test17
CPPMinimalFFT
${FFTW_LIB})

target_include_directories(test17 PRIVATE
..
)

