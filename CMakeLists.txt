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
    sincos.cpp
    wheel.cpp
    indexer.cpp
    CPPMinimalFFT.hpp
    plan.hpp
)

find_library(SYSTEM_LIB System)

target_link_libraries(CPPMinimalFFT
${SYSTEM_LIB})

add_subdirectory(test)
add_subdirectory(assembly)

