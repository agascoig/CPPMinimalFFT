# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version

project(CPPMinimalFFT CXX)          # Specify project name and language

#set(MINFFT "32") # single precision
set(MINFFT "64") # double precision

# set use SIMD code ON, or OFF for regular scalar
set(SIMD "ON")

#set(SANITIZE "-fsanitize=address, -fno-omit-frame-pointer")
set(SANITIZE "")

#set(OPTIMIZE "-O1")
set(OPTIMIZE "-O3")

set(CMAKE_CXX_FLAGS "-march=native -g ${OPTIMIZE} -std=c++23 -fPIC -DMINFFT=${MINFFT} ${SANITIZE}")

find_library(HIGHWAY_LIB highway HINT /opt/homebrew/opt/highway)

SET(LIB_SOURCES
   src/pfa.cpp
   src/plan.cpp
   src/wheel.cpp
   src/indexer.cpp
   src/sincos.cpp
   src/small.cpp
)

if(SIMD STREQUAL "ON")
  if(MINFFT STREQUAL "32")
    message(STATUS "Building SIMD single-precision")
    list(APPEND LIB_SOURCES
    src/s4/direct.cpp
    src/s4/stockham.cpp
    src/s4/bluestein.cpp)
  endif()
  if(MINFFT STREQUAL "64")
    message(STATUS "Building SIMD double-precision")
    list(APPEND LIB_SOURCES
    src/d2/direct.cpp
    src/d2/stockham.cpp
    src/d2/bluestein.cpp)
  endif()
else()
   message(STATUS "Building non-SIMD precision=${MINFFT}")
   list(APPEND LIB_SOURCES
   src/direct.cpp
   src/stockham.cpp
   src/bluestein.cpp)
endif()

add_library(CPPMinimalFFT
   ${LIB_SOURCES}
)

target_include_directories(CPPMinimalFFT PUBLIC
/usr/local/include/hwy
/opt/homebrew/opt/highway/include
src/)

add_subdirectory(test)
add_subdirectory(assembly)
