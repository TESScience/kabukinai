set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")

ADD_EXECUTABLE(psfgen main.c)
TARGET_LINK_LIBRARIES(psfgen libpsf libargparse)
