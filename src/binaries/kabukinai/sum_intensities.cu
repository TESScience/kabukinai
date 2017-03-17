#ifndef SUM_INTENSITIES_CU
#define SUM_INTENSITIES_CU
#include "star_data.h"
#include "kabukinai.h"
#include <stdio.h>
#include <stdlib.h>

// TODO: This isn't sanitary, we must use objects!
// Texture reference, local to this file
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> psf_texture;

// Call this function to make the texture from a C data array

__host__ void setup_psf_texture(const int height, const int width, const float *data) {

	// Make the type definition for a single float element
	
	const cudaChannelFormatDesc floatDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat );
	
	// Make an array to hold the texture on the device

	cudaArray* floatArray;
	PANIC_ON_BAD_CUDA_STATUS(cudaMalloc3DArray(&floatArray, &floatDesc, make_cudaExtent( height, width, STAR_COLORS ), 0));

	// Copy the data to the array
	const size_t size = height * width * STAR_COLORS * sizeof( float );
	PANIC_ON_BAD_CUDA_STATUS(cudaMemcpyToArray(floatArray, 0, 0, data, size, cudaMemcpyHostToDevice));

	// Return zero for accesses outside the PSF texture
	
	psf_texture.addressMode[0] = cudaAddressModeBorder;
	psf_texture.addressMode[1] = cudaAddressModeBorder;
	
	// Interpolate between samples
	
	psf_texture.filterMode = cudaFilterModeLinear;

	// Use [0,1] as the coordinate limits of the texture.
	// This means cu_psf need not pay attention to oversampling.
	
	psf_texture.normalized = true;

	// Bind the texture to its data
		
	PANIC_ON_BAD_CUDA_STATUS(cudaBindTextureToArray(psf_texture, floatArray, floatDesc));
}

// Get the PSF amplitude at pixel coordinates x, y relative to the 
// center of the PSF

__device__ inline float cu_psf(const float x, const float y, const int color, const star_meta_data meta_data) {
    const float norm_x = x / meta_data.single_panel_pixel_dimensions.x_dimension + 0.5;
    const float norm_y = y / meta_data.single_panel_pixel_dimensions.y_dimension + 0.5;
    return tex2DLayered(psf_texture, norm_x, norm_y, color);
}


__global__ void
sum_intensities_for_pixel(float *pixel, const star *stars, int *panel_indices, const star_meta_data meta_data) {

    float my_pixel = 0.0;   // This thread's pixel value
    const int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_y = blockIdx.y * blockDim.y + threadIdx.x;
    const int pixel_index = pixel_y * meta_data.image_dimensions.x_dimension + pixel_x;

    for (int panel_indexX = blockIdx.x - 1; panel_indexX <= blockIdx.x + 1; ++panel_indexX) {
        for (int panel_indexY = blockIdx.y - 1; panel_indexY <= blockIdx.y + 1; ++panel_indexY) {
            const int neighborhood_index = panel_index_lookup(panel_indexX, panel_indexY, meta_data);
            const int panel_start = panel_indices[neighborhood_index];
            const int panel_end = panel_indices[neighborhood_index + 1];
            for (int star_index = panel_start; star_index < panel_end; ++star_index) {
                const star star_data = stars[star_index];
                for (int color = 0; color < STAR_COLORS; ++color)
                    my_pixel +=
                            star_data.intensities[color] *
                            cu_psf(star_data.point.x - pixel_x, star_data.point.y - pixel_y, color, meta_data);
            }
        }
    }
    pixel[pixel_index] = my_pixel;
}

#endif // SUM_INTENSITIES_CU
