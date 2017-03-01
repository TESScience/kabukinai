set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")

ADD_LIBRARY(libpsfgen SHARED psf.c fits_utils.c psf.h)
TARGET_LINK_LIBRARIES(libpsfgen m cfitsio)

ADD_EXECUTABLE(psfgen_unit_tests unit_tests.c psf.h)
TARGET_LINK_LIBRARIES(psfgen_unit_tests libpsfgen)

SET(psfgen_benchmark_SOURCES benchmark.c psf.h)

ADD_EXECUTABLE(psfgen_benchmark ${psfgen_benchmark_SOURCES})
TARGET_LINK_LIBRARIES(psfgen_benchmark libpsfgen)

ADD_EXECUTABLE(psfgen main.c psf.h)
TARGET_LINK_LIBRARIES(psfgen libpsfgen libargparse)
