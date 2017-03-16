#ifndef SUM_INTENSITIES_H
#define SUM_INTENSITIES_H

#include "../../libraries/star_data/star_data.h"

__host__ void setup_psf_texture( int height, int width, float *data);

__global__ void
sum_intensities_for_pixel(float *pixel, const star *stars, int *panel_indices, const star_meta_data meta_data); 

#endif //SUM_INTENSITIES_H
