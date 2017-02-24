cmake_minimum_required(VERSION 2.8)

project(KABUKINAI)

find_package(CUDA REQUIRED)
add_subdirectory(src)

set(CMAKE_CXX_FLAG "-g -Wall")
