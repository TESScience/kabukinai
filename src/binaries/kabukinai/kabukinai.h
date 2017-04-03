#ifndef KABUKINAI_H
#define KABUKINAI_H

#include <stdio.h>
#include <stdbool.h>

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
    int number_of_slices, early_dark_pixels, late_dark_pixels, smear_rows, final_dark_rows;
    float smear_ratio, exposure_time;
    unsigned long long random_seed, random_offset;
    float *read_noise_variance;	// dimension is number_of_slices
} simulation_data;

int write_simulation_fits(simulation_data data, const char *fits_file_name, const char *history);

void inline simulation_data_release(simulation_data data) {
    free(data.image_pixels);
}


#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__

// HTTM-derived functions

__host__ void to_slices( simulation_data *d );
__host__ void expose_and_smear( simulation_data *d );
__host__ void add_noise( simulation_data *d );

#endif // __CUDACC__

#endif //KABUKINAI_H
