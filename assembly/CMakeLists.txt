# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)  # Minimum CMake version
project(asssembly CXX)          # Specify project name and language

# will not accept -march=native for assembly generation
separate_arguments(CXX_FLAGS_LIST UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
list(REMOVE_ITEM CXX_FLAGS_LIST "-march=native")

set(ASM_FLAGS
    -I/usr/local/include
    -I/opt/homebrew/opt/highway/include
    -I..
    -I../..
)

set(SRC_ASM_FILES)

foreach(f ${LIB_SOURCES})
   list(APPEND SRC_ASM_FILES "../${f}")
endforeach()

foreach(src ${SRC_ASM_FILES})

    get_filename_component(src_name ${src} NAME_WE)
    get_filename_component(src_abs ${src} ABSOLUTE)

    set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${src_name}.s")
    set(out_file_simple "${CMAKE_CURRENT_BINARY_DIR}/${src_name}_clean.s")

    add_custom_command(
        OUTPUT ${out_file}

        COMMAND ${CMAKE_CXX_COMPILER}
                ${CXX_FLAGS_LIST}
                ${ASM_FLAGS}
                -S
                ${src_abs}
                -o
                ${out_file}

        COMMAND ${CMAKE_CXX_COMPILER}
                ${CXX_FLAGS_LIST}
                ${ASM_FLAGS}
                -gline-tables-only
                -S
                ${src_abs}
                -o
                ${out_file_simple}

        DEPENDS ${src_abs}

        COMMENT "Generating assembly for ${src_abs}"
        VERBATIM
    )

    list(APPEND ASM_OUTPUTS ${out_file})

endforeach()

message(STATUS "ASM_OUTPUTS: ${ASM_OUTPUTS}")
add_custom_target(assembly ALL DEPENDS ${ASM_OUTPUTS})



