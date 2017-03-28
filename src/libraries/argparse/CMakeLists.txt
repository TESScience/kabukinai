SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-long-long -pedantic")

ADD_LIBRARY(argparse SHARED argparse.c)
SET_PROPERTY(TARGET argparse PROPERTY C_STANDARD 11)
