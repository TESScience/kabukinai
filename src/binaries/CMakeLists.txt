set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-long-long -Werror")

ADD_SUBDIRECTORY(psfgen)
ADD_SUBDIRECTORY(kabukinai)
