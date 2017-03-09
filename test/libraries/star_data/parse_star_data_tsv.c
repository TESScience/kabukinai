#include <stdio.h>
#include <stdlib.h>
#include "star_data.h"

int main(const int argc, const char* argv[]) {
    if (argc < 2) {
        printf("No input file specified\n");
        return EXIT_FAILURE;
    }
    printf("Input file: %s\n", argv[1]);
    star_data data;
    dimensions image_dimensions;
    image_dimensions.x_dimension = 2048;
    image_dimensions.y_dimension = 2058;
    dimensions single_panel_pixel_dimensions;
    single_panel_pixel_dimensions.x_dimension = 32;
    single_panel_pixel_dimensions.y_dimension = 32;
    parse_star_data_from_tsv(&data, argv[1], image_dimensions, single_panel_pixel_dimensions);
    star_data_release(data);
    return EXIT_SUCCESS;
}