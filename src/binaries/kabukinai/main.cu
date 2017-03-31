#include <stdio.h>
#include "psf.h"
#include "kabukinai.h"
#include "star_data.h"
#include "sum_intensities.cuh"
#include <stdlib.h>
#include "argparse.h"

#define PANEL_SIDE_DIMENSION 32
#define TESS_IMAGE_HEIGHT 2058
#define TESS_IMAGE_WIDTH 2048

static const char *description = "\nkabukinai is a program for generating simulated camera FITS images";

static const char *usage[] = {
        "kabukinai psf.fits star_data.tsv output.fits",
        NULL,
};

int main(int argc, const char *argv[]) {
    struct argparse_option options[] = {
            OPT_HELP(),
            OPT_END(),
    };

    struct argparse parser;
    argparse_init(&parser, options, usage, 0);
    argparse_describe(&parser, description, "");
    argc = argparse_parse(&parser, argc, argv);

    if (argc != 3) {
        fprintf(stderr, "Insufficient number of command line arguments given\n");
        exit(EXIT_FAILURE);
    }

    psf_data point_spread_function_data;
    read_psf_fits(&point_spread_function_data, argv[0]);

    setup_psf_texture((int) point_spread_function_data.dimensions[0],
                      (int) point_spread_function_data.dimensions[1],
                      point_spread_function_data.image_pixels);

    star_data star_data_from_file;
    dimensions image_dimensions;
    image_dimensions.x_dimension = TESS_IMAGE_WIDTH;
    image_dimensions.y_dimension = TESS_IMAGE_HEIGHT;

    dimensions single_panel_pixel_dimensions;
    single_panel_pixel_dimensions.x_dimension = PANEL_SIDE_DIMENSION;
    single_panel_pixel_dimensions.y_dimension = PANEL_SIDE_DIMENSION;

    parse_star_data_from_tsv(&star_data_from_file, argv[1], image_dimensions, single_panel_pixel_dimensions);


    const size_t image_size = sizeof(float) * image_dimensions.x_dimension * image_dimensions.y_dimension;

    float *pixels;
    PANIC_ON_BAD_CUDA_STATUS(cudaMalloc((void **) &pixels, image_size));
    PANIC_ON_BAD_CUDA_STATUS(cudaMemset((void *) pixels, 0, image_size));

    star_pixel_coordinate *stars;
    const size_t stars_size = sizeof(star_pixel_coordinate) * NUMBER_OF_STARS(star_data_from_file);

    PANIC_ON_BAD_CUDA_STATUS(cudaMalloc((void **) &stars, stars_size));
    PANIC_ON_BAD_CUDA_STATUS(cudaMemcpy(stars, star_data_from_file.stars, stars_size, cudaMemcpyHostToDevice));

    int *panel_indices;
    const size_t panel_indices_size = sizeof(int) * NUMBER_OF_PANEL_INDICES(star_data_from_file);

    PANIC_ON_BAD_CUDA_STATUS(cudaMalloc((void **) &panel_indices, panel_indices_size));
    PANIC_ON_BAD_CUDA_STATUS(cudaMemcpy(panel_indices,
                                        star_data_from_file.panel_indices,
                                        panel_indices_size,
                                        cudaMemcpyHostToDevice));

    dim3 dgrid(image_dimensions.y_dimension / single_panel_pixel_dimensions.y_dimension,
               image_dimensions.x_dimension / single_panel_pixel_dimensions.x_dimension);

    dim3 dblock(single_panel_pixel_dimensions.y_dimension, single_panel_pixel_dimensions.x_dimension);

    sum_intensities_for_pixel << < dgrid, dblock >> > (pixels, stars, panel_indices, star_data_from_file.meta_data);


    PANIC_ON_BAD_CUDA_STATUS(cudaDeviceSynchronize());

    /*
     * Copy the result back to the host.
     */

    simulation_data result;
    result.image_pixels = (float *) malloc(image_size);
    result.dimensions[0] = image_dimensions.x_dimension;
    result.dimensions[1] = image_dimensions.y_dimension;
    PANIC_ON_BAD_CUDA_STATUS(cudaMemcpy(result.image_pixels, pixels, image_size, cudaMemcpyDeviceToHost));


    write_simulation_fits(result, argv[2], "TODO");

    PANIC_ON_BAD_CUDA_STATUS(cudaDeviceReset());
    star_data_release(star_data_from_file);
    psf_data_release(point_spread_function_data);
    simulation_data_release(result);
    exit(EXIT_SUCCESS);
}
