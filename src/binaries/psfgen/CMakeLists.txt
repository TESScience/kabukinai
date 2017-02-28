if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()

ADD_LIBRARY(libpsfgen SHARED psf.c psf.h)
TARGET_LINK_LIBRARIES(libpsfgen m)

ADD_EXECUTABLE(psfgen_unit_tests unit_tests.c psf.h)
TARGET_LINK_LIBRARIES(psfgen_unit_tests libpsfgen)

SET(psfgen_benchmark_SOURCES benchmark.c psf.h)

ADD_EXECUTABLE(psfgen_benchmark ${psfgen_benchmark_SOURCES})
TARGET_LINK_LIBRARIES(psfgen_benchmark libpsfgen)

ADD_EXECUTABLE(psfgen main.c psf.h)
TARGET_LINK_LIBRARIES(psfgen libpsfgen libargparse cfitsio)
