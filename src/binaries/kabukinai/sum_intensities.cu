#ifndef SUM_INTENSITIES_CU
#define SUM_INTENSITIES_CU

#include "sum_intensities.cuh"
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

    const cudaChannelFormatDesc floatDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // Make an array to hold the texture on the device

    cudaArray *floatArray;
    PANIC_ON_BAD_CUDA_STATUS(
            cudaMalloc3DArray(&floatArray, &floatDesc, make_cudaExtent(height, width, STAR_COLORS), 0));

    // Copy the data to the array
    const size_t size = height * width * STAR_COLORS * sizeof(float);
    PANIC_ON_BAD_CUDA_STATUS(cudaMemcpyToArray(floatArray, 0, 0, data, size, cudaMemcpyHostToDevice));

    // Return zero for accesses outside the PSF texture

    psf_texture.addressMode[0] = cudaAddressModeBorder;
    psf_texture.addressMode[1] = cudaAddressModeBorder;

    // Interpolate between samples

    psf_texture.filterMode = cudaFilterModeLinear;

    // Use [0,1] as the coordinate limits of the texture.
    // This means CU_PSF need not pay attention to oversampling.

    psf_texture.normalized = true;

    // Bind the texture to its data

    PANIC_ON_BAD_CUDA_STATUS(cudaBindTextureToArray(psf_texture, floatArray, floatDesc));
}

// Get the PSF amplitude at pixel coordinates x, y relative to the center of the PSF
#define CU_PSF(x, y, color, meta_data, the_texture) \
    tex2DLayered((the_texture), \
                 (x) / (meta_data).single_panel_pixel_dimensions.x_dimension + 0.5, \
                 (y) / (meta_data).single_panel_pixel_dimensions.y_dimension + 0.5, \
                 (color))

__global__ void
sum_intensities_for_pixel(float *pixel, const star *stars, int *panel_indices, const star_meta_data meta_data) {

    float my_pixel = 0.0;   // This thread's pixel value
    const int pixel_x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_y_coordinate = blockIdx.y * blockDim.y + threadIdx.x;
    const float pixel_x = (float) pixel_x_coordinate + 0.5f;
    const float pixel_y = (float) pixel_y_coordinate + 0.5f;

    printf("***********pixel_x = %g, pixel_y = %g***********\n", pixel_x, pixel_y);

    for (int panel_indexX = (int) blockIdx.x - 1; panel_indexX <= (int) blockIdx.x + 1; ++panel_indexX) {
        for (int panel_indexY = (int) blockIdx.y - 1; panel_indexY <= (int) blockIdx.y + 1; ++panel_indexY) {
            // TODO: Check if panel indices are valid

            printf("*********panel_indexX = %d, panel_indexY = %d***********\n", panel_indexX, panel_indexY);

            const int neighborhood_index = PANEL_INDEX_LOOKUP_BY_PANEL_INDICES(panel_indexX, panel_indexY, meta_data);

            printf("***********neighborhood_index = %d*************\n", neighborhood_index);

            const int panel_start = panel_indices[neighborhood_index];

            printf("***********panel_start = %d*************\n", panel_start);

            const int panel_end = panel_indices[neighborhood_index + 1];

            printf("***********panel_end = %d*************\n", panel_end);

            for (int star_index = panel_start; star_index < panel_end; ++star_index) {
                const star star_data = stars[star_index];
                for (int color = 0; color < STAR_COLORS; ++color) {

                    printf("************intensity = %f***********\n", star_data.intensities[color]);
                    printf("************x = %f, y = %f***********\n", star_data.point.x, star_data.point.y);
                    printf("************pixel x = %f, pixel y = %f***********\n", pixel_x, pixel_y);
                    printf("************d x = %f, d y = %f***********\n", star_data.point.x - pixel_x, star_data.point.y - pixel_y);
                    printf("************psf = %f***********\n", CU_PSF(star_data.point.x - pixel_x,
                                                                       star_data.point.y - pixel_y,
                                                                       color,
                                                                       meta_data,
                                                                       psf_texture));

                    my_pixel += star_data.intensities[color] * CU_PSF(star_data.point.x - pixel_x,
                                                                      star_data.point.y - pixel_y,
                                                                      color,
                                                                      meta_data,
                                                                      psf_texture);
                }
            }
        }
    }

    printf("**************my_pixel = %f**************\n", my_pixel);

    const int pixel_index = pixel_y_coordinate * meta_data.image_dimensions.x_dimension + pixel_x_coordinate;
    pixel[pixel_index] = my_pixel;
}

#endif // SUM_INTENSITIES_CU
