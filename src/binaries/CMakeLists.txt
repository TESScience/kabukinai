SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Werror")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-long-long -Werror")

ADD_SUBDIRECTORY(psfgen)
ADD_SUBDIRECTORY(kabukinai)
