SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

SET(CUDA_SRCS main.cu kabukinai.h fits_utils.c psf.h star_data.h sum_intensities.cu sum_intensities.cuh argparse.h)
CUDA_ADD_EXECUTABLE(kabukinai ${CUDA_SRCS})
TARGET_LINK_LIBRARIES(kabukinai cfitsio argparse star_data psf)
