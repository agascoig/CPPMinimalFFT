# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version

project(CPPMinimalFFT CXX)          # Specify project name and language

set(VERSION "V0.1.0") # version string

#set(MINFFT "32") # single precision
set(MINFFT "64") # double precision

# set use SIMD code ON, or OFF for regular scalar
set(SIMD "ON")
#set(SIMD "OFF")

#set(SANITIZE "-fsanitize=address -fno-omit-frame-pointer")

#set(OPTIMIZE "-O0") # cannot do with Google Highway SIMD ON
#set(OPTIMIZE "-O1")
set(OPTIMIZE "-O3 -fomit-frame-pointer -fno-stack-protector -ffast-math")

set(NDEBUG "-DNDEBUG")

# remove -fPIC
set(CMAKE_CXX_FLAGS "-march=native -g ${OPTIMIZE} -std=c++23 -DMINFFT=${MINFFT} ${SANITIZE} ${NDEBUG} -DSIMD=\\\"${SIMD}\\\" -DVERSION=\\\"${VERSION}\\\"")

find_library(HIGHWAY_LIB highway HINTS /opt/homebrew/opt/highway)

SET(LIB_SOURCES
   src/pfa.cpp
   src/plan.cpp
   src/wheel.cpp
   src/indexer.cpp
)

if(SIMD STREQUAL "ON")
  if(MINFFT STREQUAL "32")
    message(STATUS "Building SIMD single-precision")
    list(APPEND LIB_SOURCES
    src/s4/direct.cpp
#    src/s4/bluestein.cpp # currently fails test2 sweep SIMD=ON
    src/bluestein.cpp
    src/s4/spiral.cpp
    src/s4/small.cpp)
  endif()
  if(MINFFT STREQUAL "64")
    message(STATUS "Building SIMD double-precision")
    list(APPEND LIB_SOURCES
    src/d2/direct.cpp
    src/d2/spiral.cpp
    src/d2/bluestein.cpp
    src/d2/small.cpp)
  endif()
else()
   message(STATUS "Building non-SIMD precision=${MINFFT}")
   list(APPEND LIB_SOURCES
   src/direct.cpp
   src/spiral.cpp
   src/bluestein.cpp
   src/small.cpp)
endif()

configure_file(
  ${CMAKE_SOURCE_DIR}/buildinfo.h.in
  ${CMAKE_BINARY_DIR}/buildinfo.h
  @ONLY)

add_library(CPPMinimalFFT
   ${LIB_SOURCES}
)

target_include_directories(CPPMinimalFFT PUBLIC
/usr/local/include/hwy
/opt/homebrew/opt/highway/include
src/)

add_subdirectory(test)
#add_subdirectory(assembly)
#add_subdirectory(testv)

#add_dependencies(CPPMinimalFFT assembly)