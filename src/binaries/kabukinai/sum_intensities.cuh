#ifndef SUM_INTENSITIES_CUH
#define SUM_INTENSITIES_CUH

#include "star_data.h"

#ifdef __cplusplus
extern "C" {
#endif

__host__ void setup_psf_texture(const int height, const int width, const float *data);

__global__ void
sum_intensities_for_pixel(float *pixel, const star_pixel_coordinate *stars, int *panel_indices, const star_meta_data meta_data);

#ifdef __cplusplus
}
#endif

#endif // SUM_INTENSITIES_CUH
