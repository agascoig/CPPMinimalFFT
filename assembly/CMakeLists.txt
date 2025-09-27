# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version
project(asssembly CXX)          # Specify project name and language

set(ASM_FLAGS
    -march=native
    -g
    -O3
    -fPIC
    -std=c++23
    -I/usr/local/include
    -I..
)

set(SRC_FILES
    ../direct.cpp
    ../bluestein.cpp
    ../pfa.cpp
    ../plan.cpp
    ../stockham.cpp
    ../wheel.cpp
    ../indexer.cpp
    ../sincos.cpp
)

foreach(src ${SRC_FILES})
    get_filename_component(src_name ${src} NAME_WE)
    get_filename_component(src_abs ${src} ABSOLUTE)
    add_custom_command(
        OUTPUT ${src_name}.s
        COMMAND ${CMAKE_CXX_COMPILER} ${ASM_FLAGS} -S ${src_abs} -o ${src_name}.s > ${src_name}_build.txt 2>&1
        DEPENDS ${src_abs}
        COMMENT "Generating assembly for ${src_abs}"
        VERBATIM
    )
    list(APPEND ASM_OUTPUTS ${src_name}.s)
endforeach()

add_custom_target(assembly ALL DEPENDS ${ASM_OUTPUTS})



