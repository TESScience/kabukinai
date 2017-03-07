#include "../../libraries/star_data/star_data.h"
#include "stdbool.h"

 
/* 
 * Get the PSF amplitude at the thread's pixel for the star at x, y.
 */
__device__ inline float cu_psf( float x, float y, int color ) {
	return 0.0;	/* STUB */
}



__global__ void sum_intensities_for_pixel(float *pixel, const star *stars, const dimensions panel_indices_dimensions, int *panel_indices, const dimensions image_dimensions) {

    float my_pixel = 0.0;   /* This thread's pixel value */
    const int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_y = blockIdx.y * blockDim.y + threadIdx.x;
    const int pixel_coordinate = pixel_x * image_dimensions.y_dimension + pixel_y;

    for(int panel_indexX = blockIdx.x - 1; panel_indexX <= blockIdx.x + 1 ; ++panel_indexX) {
      for(int panel_indexY = blockIdx.y - 1; panel_indexY <= blockIdx.y + 1 ; ++panel_indexY) {
       const int neighborhood_index = panel_index_lookup(panel_indexX, panel_indexY, panel_indices_dimensions);
       const int panel_start = panel_indices[neighborhood_index];
       const int panel_end = panel_indices[neighborhood_index+1];
       for (int star_index = panel_start; star_index < panel_end; ++star_index) {
         const star star_data = stars[star_index];
         for (int color = 0 ; color < STAR_COLORS; ++color) 
// TODO add layered texture
//            my_pixel +=  star_data.intensities[color] * cu_psf(star_data.x - pixel_x, star_data.y - pixel_y, color, psf_texture);
            my_pixel +=  star_data.intensities[color] * cu_psf(star_data.x - pixel_x, star_data.y - pixel_y, color);
       }
     }
	}
     pixel[pixel_coordinate] = my_pixel;
}
