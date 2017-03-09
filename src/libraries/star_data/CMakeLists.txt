SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")

SET(libstardata_SOURCES parse_star_data_from_tsv.cpp star_data.c star_data.h)
ADD_LIBRARY(libstar_data SHARED ${libstardata_SOURCES})
TARGET_COMPILE_FEATURES(libstar_data PRIVATE cxx_range_for)
