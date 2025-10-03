# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version

project(CPPMinimalFFT CXX)          # Specify project name and language

set(CMAKE_CXX_FLAGS "-march=native -g -O3 -std=c++23 -fPIC")

add_library(CPPMinimalFFT
    direct.cpp
    bluestein.cpp
    pfa.cpp
    plan.cpp
    stockham.cpp
    wheel.cpp
    indexer.cpp
    sincos.cpp
)

target_include_directories(CPPMinimalFFT PRIVATE
.
)

add_subdirectory(test)
add_subdirectory(assembly)

