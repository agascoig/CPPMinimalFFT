# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version
project(asssembly CXX)          # Specify project name and language

set(ASM_FLAGS
    -march=native
    -g
    -O3
    -fPIC
#    -fverbose-asm
    -std=c++23
#    -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.0.sdk
    -I/usr/local/include
    -I/opt/homebrew/opt/highway/include
    -I..
    -I../..
)

set(SRC_FILES
#    ../src/direct.cpp
#    ../src/stockham.cpp
#    ../src/bluestein.cpp
    ../src/d2/direct.cpp
    ../src/d2/bluestein.cpp
    ../src/d2/stockham.cpp
    ../src/pfa.cpp
    ../src/plan.cpp
    ../src/wheel.cpp
    ../src/indexer.cpp
    ../src/sincos.cpp
)

foreach(src ${SRC_FILES})
    get_filename_component(src_name ${src} NAME_WE)
    get_filename_component(src_abs ${src} ABSOLUTE)
    add_custom_command(
        OUTPUT ${src_name}.s
        COMMAND ${CMAKE_CXX_COMPILER} ${ASM_FLAGS} -v -S ${src_abs} -o ${src_name}.s > ${src_name}_build.txt 2>&1
        DEPENDS ${src_abs}
        COMMENT "Generating assembly for ${src_abs}"
        VERBATIM
    )
    list(APPEND ASM_OUTPUTS ${src_name}.s)
endforeach()

add_custom_target(assembly ALL DEPENDS ${ASM_OUTPUTS})



