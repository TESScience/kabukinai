ADD_LIBRARY(libpsfgen SHARED psf.c psf.h)
TARGET_LINK_LIBRARIES(libpsfgen m)

ADD_EXECUTABLE(psfgen_unit_tests unit_tests.c psf.h)
TARGET_LINK_LIBRARIES(psfgen_unit_tests libpsfgen)

SET(psfgen_benchmark_SOURCES benchmark.c psf.h)

ADD_EXECUTABLE(psfgen_benchmark ${psfgen_benchmark_SOURCES})
TARGET_LINK_LIBRARIES(psfgen_benchmark libpsfgen)
