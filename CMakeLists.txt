# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version

project(CPPMinimalFFT CXX)          # Specify project name and language

set(CMAKE_CXX_FLAGS "-march=native -g -O3 -std=c++23 -fPIC")

add_library(CPPMinimalFFT
    src/direct.cpp
    src/bluestein.cpp
    src/pfa.cpp
    src/plan.cpp
    src/stockham.cpp
    src/wheel.cpp
    src/indexer.cpp
    src/sincos.cpp
)

target_include_directories(CPPMinimalFFT PRIVATE
src/
)

add_subdirectory(test)
add_subdirectory(assembly)

