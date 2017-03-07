CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)
PROJECT(Kabukinai)

enable_testing()
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

ADD_SUBDIRECTORY(libraries)
ADD_SUBDIRECTORY(binaries)

