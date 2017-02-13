# Append current NVCC flags by something, eg comput capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

set(CUDASRCS main.cu)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set(CMAKE_CXX_FLAG "-g -Wall")
CUDA_ADD_EXECUTABLE(kabukinai ${CUDASRCS})
