set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")

ADD_LIBRARY(libpsf SHARED psf.c fits_utils.c psf_data.c psf.h)
TARGET_LINK_LIBRARIES(libpsf m cfitsio)

ADD_EXECUTABLE(psf_unit_tests unit_tests.c psf.h)
TARGET_LINK_LIBRARIES(psf_unit_tests libpsf)
ADD_TEST(psf_test psf_unit_tests)
ADD_DEPENDENCIES(check psf_unit_tests)

SET(psf_benchmark_SOURCES benchmark.c psf.h)

ADD_EXECUTABLE(psf_benchmark ${psf_benchmark_SOURCES})
TARGET_LINK_LIBRARIES(psf_benchmark libpsf)
