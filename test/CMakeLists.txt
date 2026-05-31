# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version

project(test17 CXX)          # Specify project name and language

if (SIMD STREQUAL ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSIMDON")
endif()

add_executable(test17
    test17.cpp
    hmean.cpp
)

add_executable(test18
    test18.cpp
    hmean.cpp
)

target_compile_options(test17 PRIVATE ${ASAN_FLAGS})
target_link_options(test17 PRIVATE ${ASAN_FLAGS})

find_library(FFTW_LIB fftw3 HINT /usr/lib/aarch64-linux-gnu)
find_library(FFTWF_LIB fftw3f HINT /usr/lib/aarch64-linux-gnu)

#set(FFTW_LIB "/Users/alexandergascoigne/fft/fftw-install/lib/libfftw3.dylib")

set(COMMON_LIBS PRIVATE
CPPMinimalFFT
${FFTW_LIB}
${FFTWF_LIB})

set(COMMON_INCLUDES PRIVATE
${CMAKE_BINARY_DIR}
../src/
.
/Users/alexandergascoigne/fft/fftw-install/include
/opt/homebrew/include
)

target_include_directories(test17 PRIVATE ${COMMON_INCLUDES})
target_include_directories(test18 PRIVATE ${COMMON_INCLUDES})

target_link_libraries(test17 PRIVATE ${COMMON_LIBS})
target_link_libraries(test18 PRIVATE ${COMMON_LIBS})

