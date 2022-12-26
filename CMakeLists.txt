cmake_minimum_required(VERSION 3.16)
project(lab2)


set(CMAKE_CXX_STANDARD 20)

add_executable(generator source.cpp thread_num.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++2a -m64")