#include "../../libraries/star_data/star_data.h"
#include "stdbool.h"

/*
 * A block is associated with the panels that contain stars that
 * can influence it. If there are, e.g., 9 such panels, threads in
 * the block can access them by numbers 0-8. The functions below figure
 * this out using blockIdx.x and blockIdx.y to identify which block they
 * are serving.
 */

/*
 * Indicate whether the block is has an associated panel numbered i.
 */

__device__ inline bool block_has_panel( int i ){
	return false;	/* STUB */
}

/*
 * Return the number of stars in panel i.
 */

__device__ inline int number_of_stars_in_panel( int i ){
	return 0;	/* STUB */
}

/*
 * Return a pointer to the stars in panel i.
 */

__device__ inline star *array_of_stars_in_panel( int i ){
	return NULL;	/* STUB */
}

/*
 * The following use blockIdx and threadIdx to figure out which
 * pixel the thread represents.
 */
 
/* 
 * Get the PSF amplitude at the thread's pixel for the star at x, y.
 */

__device__ inline float cu_psf( float x, float y, int color ) {
	return 0.0;	/* STUB */
}

/*
 * Set the thread's pixel value in the output image.
 */

__device__ inline void set_pixel_in_raster( float my_pixel ){
	return;		/* STUB */
}


__global__ void sum_intensities_for_pixel( ) {
	
	float my_pixel = 0.0;	/* This thread's pixel value */
	int i, j, color;

	for( i = 0; block_has_panel( i ); i += 1){
		int number_of_stars = number_of_stars_in_panel( i );
		star * starp = array_of_stars_in_panel( i );
		for( j = 0; j < number_of_stars; j += 1 ){
			for( color = 0; color < STAR_COLORS; color += 1){
				my_pixel += starp->intensities[color] *
					cu_psf( starp->x, starp->y, color );
			}
		}
	}
	set_pixel_in_raster( my_pixel );
}
