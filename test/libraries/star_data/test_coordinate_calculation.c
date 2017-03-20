#include <stdio.h>
#include <stdlib.h>
#include "star_data.h"

#define TEST_PANEL_INDEX_LOOKUP(x, y, meta_data, expected_index) { \
    if (!CHECK_PANEL_INDICES_VALID(x, y, meta_data)) { \
        fprintf(stderr, "Panel coordinates (%i, %i) are invalid\n", x, y); \
        exit(EXIT_FAILURE); \
    } \
    int panel_index = PANEL_INDEX_LOOKUP_BY_PANEL_INDICES(x, y, meta_data); \
    if (panel_index != expected_index) { \
        fprintf(stderr, "Index of panel coordinates (%i, %i) should be %i, was %i\n", \
                x, y, expected_index, panel_index); \
        exit(EXIT_FAILURE); \
    } \
}

#define TEST_PIXEL_LOOKUP(x, y, meta_data, expected_index) { \
    if (!CHECK_PIXEL_VALID(x, y, meta_data)) { \
        fprintf(stderr, "Pixel coordinates (%i, %i) are invalid\n", x, y); \
        exit(EXIT_FAILURE); \
    } \
    int panel_index = PANEL_INDEX_LOOKUP_BY_PIXEL(x, y, meta_data); \
    if (panel_index != expected_index) { \
        fprintf(stderr, "Panel index of pixel coordinates (%i, %i) should be %i, was %i\n", \
                x, y, expected_index, panel_index); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    star_meta_data meta_data;
    meta_data.single_panel_pixel_dimensions.x_dimension = 32;
    meta_data.single_panel_pixel_dimensions.y_dimension = 32;
    meta_data.panel_indices_dimensions.x_dimension = 66;
    meta_data.panel_indices_dimensions.y_dimension = 67;
    TEST_PANEL_INDEX_LOOKUP(-1, -1, meta_data, 0);
    TEST_PANEL_INDEX_LOOKUP(0, -1, meta_data, 1);
    TEST_PIXEL_LOOKUP(-1, -1, meta_data, 0);
    exit(EXIT_SUCCESS);
}
