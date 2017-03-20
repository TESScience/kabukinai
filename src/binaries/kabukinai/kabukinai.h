#ifndef KABUKINAI_H
#define KABUKINAI_H

#include <stdio.h>

#define KABUKINAI_SUCCESS 0
#define KABUKINAI_FAILURE 1

#define PANIC_ON_BAD_CUDA_STATUS(value) { \
    const cudaError_t _m_cudaStat = value; \
        if (_m_cudaStat != cudaSuccess) { \
            fprintf(stderr, "KABUKINAI_CUDA_FAILURE: %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(KABUKINAI_FAILURE); } }

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *image_pixels;
    long dimensions[2];
} simulation_data;

int write_simulation_fits(simulation_data data, const char *fits_file_name, const char *history);

void inline simulation_data_release(simulation_data data) {
    free(data.image_pixels);
}

#ifdef __cplusplus
}
#endif

#endif //KABUKINAI_H
