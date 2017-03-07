SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

SET(LIBSTARDATA_SRCS star_data.h star_data.c parse_star_data_from_tsv.cpp)


SET(CUDA_SRCS main.cu star_data.h star_data.c sum_intensities.cu parse_star_data_from_tsv.cpp)
CUDA_ADD_EXECUTABLE(kabukinai ${CUDA_SRCS})
TARGET_LINK_LIBRARIES(kabukinai cfitsio libargparse)
target_compile_features(kabukinai PRIVATE cxx_range_for)