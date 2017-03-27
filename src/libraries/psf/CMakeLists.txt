SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")

SET(psf_SOURCES psf.c fits_utils.c psf_data.c psf.h)
ADD_LIBRARY(psf SHARED ${psf_SOURCES})
TARGET_LINK_LIBRARIES(psf m cfitsio)
SET_PROPERTY(TARGET psf PROPERTY C_STANDARD 11)
