# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version
project(genweights CXX)          # Specify project name and language

set(CMAKE_CXX_FLAGS "-march=native -g -O0 -std=c++23 -fPIC")

add_executable(genweights
    genweights.cpp
)

# Look for GMP
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIB NAMES gmp)

# Look for MPFR
find_path(MPFR_INCLUDE_DIR mpfr.h)
find_library(MPFR_LIB NAMES mpfr)

# Report if missing
if (NOT GMP_INCLUDE_DIR OR NOT GMP_LIB)
    message(FATAL_ERROR "GMP not found")
endif()

if (NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIB)
    message(FATAL_ERROR "MPFR not found")
endif()

# Create imported targets
add_library(GMP::GMP UNKNOWN IMPORTED)
set_target_properties(GMP::GMP PROPERTIES
    IMPORTED_LOCATION "${GMP_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
)

add_library(MPFR::MPFR UNKNOWN IMPORTED)
set_target_properties(MPFR::MPFR PROPERTIES
    IMPORTED_LOCATION "${MPFR_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MPFR_INCLUDE_DIR}"
)

set(ASAN_FLAGS
-v
#   -fsanitize=address
#   -fno-omit-frame-pointer
)

target_compile_options(genweights PRIVATE ${ASAN_FLAGS})
target_link_options(genweights PRIVATE ${ASAN_FLAGS})

target_link_libraries(genweights
GMP::GMP
MPFR::MPFR)

target_include_directories(genweights PRIVATE
..
${MPFR_INCLUDE_DIRS}
)

