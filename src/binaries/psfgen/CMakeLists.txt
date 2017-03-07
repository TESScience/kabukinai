set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")

ADD_EXECUTABLE(psfgen main.c)
TARGET_LINK_LIBRARIES(psfgen libpsf libargparse)
