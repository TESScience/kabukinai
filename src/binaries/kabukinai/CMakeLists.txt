SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

SET(KABUKINAI_SOURCES main.cu kabukinai.h fits_utils.c psf.h star_data.h sum_intensities.cu sum_intensities.cuh argparse.h)
CUDA_ADD_EXECUTABLE(kabukinai ${KABUKINAI_SOURCES})
TARGET_LINK_LIBRARIES(kabukinai cfitsio argparse star_data psf)
