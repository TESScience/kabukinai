SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")

SET(libpsf_SOURCES psf.c fits_utils.c psf_data.c psf.h)
ADD_LIBRARY(libpsf SHARED ${libpsf_SOURCES})
TARGET_LINK_LIBRARIES(libpsf m cfitsio)

