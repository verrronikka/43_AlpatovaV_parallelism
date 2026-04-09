cmake_minimum_required(VERSION 3.14)
project(Task3_Parallel_Solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(3 3.cpp)

target_link_libraries(3 PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(3 PRIVATE -O3 -march=native)

set_target_properties(3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
