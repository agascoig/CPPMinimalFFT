# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version

project(test17 CXX)          # Specify project name and language

set(ASAN_FLAGS "")#-fsanitize=address") #-fsanitize=address")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")

add_executable(test17
    test17.cpp
    hmean.cpp
)

target_compile_options(test17 PRIVATE ${ASAN_FLAGS})
target_link_options(test17 PRIVATE ${ASAN_FLAGS})

find_library(FFTW_LIB fftw3)
find_library(FFTWF_LIB fftw3f)

#set(FFTW_LIB "/Users/alexandergascoigne/fft/fftw-install/lib/libfftw3.dylib")

target_link_libraries(test17
CPPMinimalFFT
${FFTW_LIB}
${FFTWF_LIB})

target_include_directories(test17 PRIVATE
../src/
.
/Users/alexandergascoigne/fft/fftw-install/include
/opt/homebrew/include
)

