cmake_minimum_required(VERSION 3.10)
project(integrate C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(integrate 2.c)

target_link_libraries(integrate OpenMP::OpenMP_C m)

target_compile_options(integrate PRIVATE -Wall -O2)

message(STATUS "OpenMP found: ${OpenMP_C_FOUND}")
message(STATUS "OpenMP C flags: ${OpenMP_C_FLAGS}")