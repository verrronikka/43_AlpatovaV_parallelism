cmake_minimum_required(VERSION 3.10)
project(Task1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
add_executable(test.out 1.cpp)
target_link_libraries(test.out Threads::Threads)

set_target_properties(test.out PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)