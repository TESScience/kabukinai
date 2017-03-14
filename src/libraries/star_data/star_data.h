#ifndef KABUKINAI_STAR_DATA_H
#define KABUKINAI_STAR_DATA_H

#define STAR_COLORS 1

#ifdef __cplusplus
extern "C" {
#endif

#define KABUKINAI_STAR_DATA_SUCCESS 0

typedef struct {
    float x, y;
} point_2d;

/*
 * A star has x, y coordinates and an intensity for each star color.
 */
typedef struct {
    point_2d point;
    float intensities[STAR_COLORS];
} star;

typedef struct {
    int x_dimension, y_dimension;
} dimensions;

typedef struct {
    dimensions single_panel_pixel_dimensions;
    dimensions panel_indices_dimensions;
    dimensions image_dimensions;
} star_meta_data;

#ifdef __NVCC__
__host__ __device__
#endif

inline point_2d compute_panel_2d_indices(const float x, const float y, const star_meta_data meta_data) {
    point_2d point;
    point.x = x / meta_data.single_panel_pixel_dimensions.x_dimension + 1;
    point.y = y / meta_data.single_panel_pixel_dimensions.y_dimension + 1;
    return point;
}

#ifdef __NVCC__
__host__ __device__
#endif

inline int panel_index_lookup(const float x, const float y, const star_meta_data meta_data) {
    const point_2d panel_indices = compute_panel_2d_indices(x, y, meta_data);
    if (panel_indices.x < 0 || panel_indices.x >= meta_data.panel_indices_dimensions.x_dimension ||
        panel_indices.y < 0 || panel_indices.y >= meta_data.panel_indices_dimensions.y_dimension)
        return -1;
    return ((int) panel_indices.y) * meta_data.panel_indices_dimensions.x_dimension + ((int) panel_indices.x);
}

/*
 * Star data has an array of stars, organized into groups of panels, and an array of panel indices
 */
typedef struct {
    star *stars;
    /* panel_indices has (panel_indices_dimensions[0] * panel_indices_dimensions[1] + 1) elements
     * the final element is number of elements in the star array in data. */
    long *panel_indices;
    star_meta_data meta_data;
} star_data;

int parse_star_data_from_tsv(star_data *data,
                             const char *file_name,
                             const dimensions image_dimensions,
                             const dimensions single_panel_pixel_dimensions);

void star_data_release(star_data data);

#ifdef __cplusplus
}
#endif

#endif //KABUKINAI_STAR_DATA_H
