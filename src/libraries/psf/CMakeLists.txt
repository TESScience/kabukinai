set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")

ADD_LIBRARY(libpsf SHARED psf.c fits_utils.c psf_data.c psf.h)
TARGET_LINK_LIBRARIES(libpsf m cfitsio)

