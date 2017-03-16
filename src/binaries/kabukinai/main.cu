#include <stdio.h>
#include "psf.h"
#include "sum_intensities.cuh"
#include "kabukinai.h"
#include <stdlib.h>

int main(const int argc, char * argv[])
{
	cudaError_t code;
	psf_data point_spread_function_data;
        read_psf_fits(&point_spread_function_data, argv[1]);

	// __host__ void setup_psf_texture( int height, int width, float *data);
	// __global__ void sum_intensities_for_pixel(float *pixel, const star *stars, int *panel_indices, const star_meta_data meta_data);
        setup_psf_texture(point_spread_function_data.dimensions[0], 
			  point_spread_function_data.dimensions[1], 
			  point_spread_function_data.image_pixels);

// /*
//  * Run the kernel. Note that we don't have to tell it sizes
//  * of things in the args, as those are implied by the block dimensions.
//  */
// 	dim3 blocks_dimension( height, width );
// 	sum_intensities<<< 1, blocks_dimension  >>>
// 		( output );
// 	code = cudaDeviceSynchronize();
// 	if( code ) {
// 		printf( "cudaDeviceSynchronize: %s\n",
// 			cudaGetErrorString(code));
// 		exit(EXIT_FAILURE);
// 	}
// 
// /*
//  * Copy the result back to the host.
//  */
// 
// 	float result[height][width];
// 	code = cudaMemcpy(result, output, size, cudaMemcpyDeviceToHost);
// 	if( code ){
// 		printf( "cudaMemcpyDeviceToHost: %s\n",
// 			cudaGetErrorString(code));
// 		exit(EXIT_FAILURE);
// 	}
// /*
//  * Print the result.
//  */
// 	
// 	for ( int y =0 ; y<height ; y++ ) {
// 			for ( int x = 0 ; x<width; x++ ){
// 			printf( "%10g ", result[y][x]);
// 		}
// 		printf( "\n");
// 	}
// 
// /*
//  * If you're really done, you can tidy up with a bulldozer ;-)
//  */
// 	
// 	code = cudaDeviceReset();
// 	if( code ) {
// 		printf( "cudaMemcpyDeviceToHost: %s\n",
// 			cudaGetErrorString(code));
// 		exit(EXIT_FAILURE);
// 	}
	psf_data_release(point_spread_function_data);
	exit(EXIT_SUCCESS);
}
