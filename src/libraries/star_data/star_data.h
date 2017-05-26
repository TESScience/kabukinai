#ifndef KABUKINAI_STAR_DATA_H
#define KABUKINAI_STAR_DATA_H

#include <math.h>

#define STAR_COLORS 1

#ifdef __cplusplus
extern "C" {
#endif

#define KABUKINAI_STAR_DATA_SUCCESS 0

/*
 * 2D pixel coordinates
 */
typedef struct {
    float x, y;
} point_2d;

/*
 * 3D coordinates
 */
typedef struct {
    double x, y, z;
} point_3d;

/*
 * A star using pixel coordinates has x and y coordinates and an intensity for each star color
 */
typedef struct {
    point_2d point;
    float intensities[STAR_COLORS];
} star_pixel_coordinate;

/*
 * Sky coordinates
 */
typedef struct {
    double ra, dec;
} sky_coordinate;

/*
 * A star using sky coordinates has a right ascension and declination and an intensity for each star color
 */
typedef struct {
    sky_coordinate coordinate;
    float intensities[STAR_COLORS];
} star_sky_coordinate;

/*
 * Quaternions represent 3D rotations
 */
typedef struct {
    double w, i, j, k;
} quaternion;

typedef struct {
    int x_dimension, y_dimension;
} dimensions;

typedef struct {
    dimensions single_panel_pixel_dimensions;
    dimensions panel_indices_dimensions;
    dimensions image_dimensions;
} star_meta_data;

#define PANEL_INDEX_LOOKUP_BY_PANEL_INDICES(x, y, meta_data) \
    (((y)+1) * meta_data.panel_indices_dimensions.x_dimension + ((x)+1))
#define PANEL_COORDINATE_FOR_PIXEL_X_COORDINATE(x, metadata) \
    ((int) floorf(((float) (x)) / ((float) meta_data.single_panel_pixel_dimensions.x_dimension)))
#define PANEL_COORDINATE_FOR_PIXEL_Y_COORDINATE(y, metadata) \
    ((int) floorf(((float) (y)) / ((float) meta_data.single_panel_pixel_dimensions.y_dimension)))
#define PANEL_INDEX_LOOKUP_BY_PIXEL(x, y, meta_data) \
    PANEL_INDEX_LOOKUP_BY_PANEL_INDICES( \
        PANEL_COORDINATE_FOR_PIXEL_X_COORDINATE(x, metadata), \
        PANEL_COORDINATE_FOR_PIXEL_Y_COORDINATE(y, metadata), \
        meta_data)
#define CHECK_PANEL_INDICES_VALID(x, y, meta_data) \
    !(((x)+1) < 0 || \
      ((x)+1) >= meta_data.panel_indices_dimensions.x_dimension || \
      ((y)+1) < 0 || \
      ((y)+1) >= meta_data.panel_indices_dimensions.y_dimension)
#define CHECK_PIXEL_VALID(x, y, meta_data) \
    CHECK_PANEL_INDICES_VALID( \
        PANEL_COORDINATE_FOR_PIXEL_X_COORDINATE(x, meta_data), \
        PANEL_COORDINATE_FOR_PIXEL_Y_COORDINATE(y, meta_data), \
        meta_data)

/*
 * Star data has an array of stars, organized into groups of panels, and an array of panel indices
 */
typedef struct {
    star_pixel_coordinate *stars;

    /* panel_indices has (panel_indices_dimensions[0] * panel_indices_dimensions[1] + 1) elements
     * the final element is number of elements in the star array in data. */
    int *panel_indices;


    star_meta_data meta_data;
} star_data;

#define NUMBER_OF_PANEL_INDICES(data)  \
    ((data).meta_data.panel_indices_dimensions.x_dimension * (data).meta_data.panel_indices_dimensions.y_dimension + 1)

#define NUMBER_OF_STARS(data) ((data).panel_indices[NUMBER_OF_PANEL_INDICES(data) - 1])

int parse_star_data_from_tsv(star_data *data,
                             const char *file_name,
                             const dimensions image_dimensions,
                             const dimensions single_panel_pixel_dimensions);

void star_data_release(star_data data);

#ifdef __cplusplus
}
#endif

#endif //KABUKINAI_STAR_DATA_H
