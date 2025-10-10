# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version

project(CPPMinimalFFT CXX)          # Specify project name and language

# -DMINFFT=32 for float
set(CMAKE_CXX_FLAGS "-march=native -g -O3 -std=c++23 -fPIC")

find_library(HIGHWAY_LIB highway HINT /opt/homebrew/opt/highway)

add_library(CPPMinimalFFT
#    src/direct.cpp
#    src/stockham.cpp
#    src/bluestein.cpp
    src/d2/direct.cpp
    src/d2/stockham.cpp
    src/d2/bluestein.cpp
#    src/s4/direct.cpp
#    src/s4/stockham.cpp
#    src/s4/bluestein.cpp
    src/pfa.cpp
    src/plan.cpp
    src/wheel.cpp
    src/indexer.cpp
    src/sincos.cpp
)

target_include_directories(CPPMinimalFFT PUBLIC
/usr/local/include/hwy
/opt/homebrew/opt/highway/include
src/)

add_subdirectory(test)
add_subdirectory(assembly)
