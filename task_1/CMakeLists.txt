cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(my-proj VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_DOUBLE "Use double type in the array instead float" OFF)

if(USE_DOUBLE)
    add_definitions(-DUSE_DOUBLE)
endif()


add_executable(myexec task1.cpp)