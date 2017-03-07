cmake_minimum_required(VERSION 2.8)

project(KABUKINAI)

find_package(CUDA REQUIRED)

enable_testing()
add_subdirectory(src)
add_subdirectory(test)
