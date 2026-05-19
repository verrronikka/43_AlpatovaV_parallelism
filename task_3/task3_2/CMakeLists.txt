cmake_minimum_required(VERSION 3.10)
project(Task2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
add_executable(task2.out 2.cpp)
target_link_libraries(task2.out Threads::Threads)

set_target_properties(task2.out PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)