#include <stdio.h>
#include <stdlib.h>
#include "star_data.h"

int count_lines(const char* file_name) {
    FILE * fp = fopen(file_name, "r");
    int line_count = 0;
    char * line = NULL;
    size_t  len = 0;
    while (getline(&line, &len, fp) != -1) ++line_count;
    fclose(fp);
    return line_count;
}

int main(const int argc, const char *argv[]) {
    if (argc < 2) {
        printf("No input file specified\n");
        exit(EXIT_FAILURE);
    }
    printf("Input file: %s\n", argv[1]);
    const int line_count = count_lines(argv[1]);
    printf("Lines in file: %i\n", line_count);
    star_data data;
    dimensions image_dimensions;
    image_dimensions.x_dimension = 2048;
    image_dimensions.y_dimension = 2058;
    dimensions single_panel_pixel_dimensions;
    single_panel_pixel_dimensions.x_dimension = 32;
    single_panel_pixel_dimensions.y_dimension = 32;
    parse_star_data_from_tsv(&data, argv[1], image_dimensions, single_panel_pixel_dimensions);
    printf("Panel dimensions: %i by %i\n",
           data.meta_data.panel_indices_dimensions.x_dimension,
           data.meta_data.panel_indices_dimensions.y_dimension);
    if (NUMBER_OF_STARS(data) != line_count) {
        fprintf(stderr, "Expected number of stars in data to be %i, was %i\n", line_count, NUMBER_OF_STARS(data));
        star_data_release(data);
        exit(EXIT_FAILURE);
    }
    star_data_release(data);
    exit(EXIT_SUCCESS);
}