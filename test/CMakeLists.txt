SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic")

ADD_SUBDIRECTORY(libraries)
ADD_SUBDIRECTORY(binaries)
