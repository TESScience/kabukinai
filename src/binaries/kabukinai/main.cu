#include <stdio.h>
#include "psf.h"
#include "sum_intensities.cu"
#include "kabukinai.h"
#include "star_data.h"
#include <stdlib.h>
#include <stddef.h>

#define PANEL_SIDE_DIMENSION 32
#define TESS_IMAGE_HEIGHT 2058
#define TESS_IMAGE_WIDTH 2048

int main(const int argc, char * argv[])
{
	psf_data point_spread_function_data;
        read_psf_fits(&point_spread_function_data, argv[1]);

        setup_psf_texture(point_spread_function_data.dimensions[0], 
			  point_spread_function_data.dimensions[1], 
			  point_spread_function_data.image_pixels);

        star_data star_data_from_file;
        dimensions image_dimensions;
        image_dimensions.x_dimension = TESS_IMAGE_WIDTH;
        image_dimensions.y_dimension = TESS_IMAGE_HEIGHT;

	dimensions single_panel_pixel_dimensions;
        single_panel_pixel_dimensions.x_dimension = PANEL_SIDE_DIMENSION;
        single_panel_pixel_dimensions.y_dimension = PANEL_SIDE_DIMENSION;

	parse_star_data_from_tsv(&star_data_from_file, argv[2], image_dimensions, single_panel_pixel_dimensions);

	const size_t image_size = sizeof(float) * image_dimensions.x_dimension * image_dimensions.y_dimension;

	float * pixels;
	PANIC_ON_BAD_CUDA_STATUS(cudaMalloc((void**)&pixels, image_size));
        PANIC_ON_BAD_CUDA_STATUS(cudaMemset((void**)&pixels, 0, image_size));
        
	star * stars;
	const size_t stars_size = sizeof(star) * number_of_stars(star_data_from_file);
	PANIC_ON_BAD_CUDA_STATUS(cudaMalloc((void**)&stars, stars_size));
	PANIC_ON_BAD_CUDA_STATUS(cudaMemcpy(stars, star_data_from_file.stars, stars_size, cudaMemcpyHostToDevice));

	int * panel_indices;
	const size_t panel_indices_size = sizeof(int) * number_of_panel_indices(star_data_from_file);
	PANIC_ON_BAD_CUDA_STATUS(cudaMalloc((void**)&panel_indices, panel_indices_size));
	PANIC_ON_BAD_CUDA_STATUS(cudaMemcpy(
				panel_indices, 
				star_data_from_file.panel_indices, 
				panel_indices_size, 
				cudaMemcpyHostToDevice));

	dim3 dgrid(image_dimensions.y_dimension/single_panel_pixel_dimensions.y_dimension,
			image_dimensions.x_dimension/single_panel_pixel_dimensions.x_dimension);
	dim3 dblock(single_panel_pixel_dimensions.y_dimension, single_panel_pixel_dimensions.x_dimension);
	sum_intensities_for_pixel<<<dgrid, dblock>>>(pixels, stars, panel_indices, star_data_from_file.meta_data);


 	PANIC_ON_BAD_CUDA_STATUS(cudaDeviceSynchronize());
 
        /*
         * Copy the result back to the host.
         */

	simulation_data result;
	result.image_pixels = (float *) malloc(image_size);
        result.dimensions[0] = image_dimensions.x_dimension;
	result.dimensions[1] = image_dimensions.y_dimension;
	PANIC_ON_BAD_CUDA_STATUS(cudaMemcpy(result.image_pixels, pixels, image_size, cudaMemcpyDeviceToHost));

        
        write_simulation_fits(result, argv[3], "TODO"); 
 	
 	PANIC_ON_BAD_CUDA_STATUS(cudaDeviceReset());
	psf_data_release(point_spread_function_data);
	simulation_data_release(result);
	exit(EXIT_SUCCESS);
}
