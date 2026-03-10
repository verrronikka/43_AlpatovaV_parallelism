cmake_minimum_required(VERSION 3.10)
project(matrix_vector_product C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matrix_mv 1.c)

target_link_libraries(matrix_mv OpenMP::OpenMP_C)

target_compile_options(matrix_mv PRIVATE -Wall -O2)

message(STATUS "OpenMP found: ${OpenMP_C_FOUND}")
message(STATUS "OpenMP C flags: ${OpenMP_C_FLAGS}")