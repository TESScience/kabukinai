#ifndef KABUKINAI_STAR_DATA_H
#define KABUKINAI_STAR_DATA_H

#define STAR_COLORS 1

#ifdef __cplusplus
extern "C" {
#endif

#define KABUKINAI_STAR_DATA_SUCCESS 0
#define KABUKINAI_STAR_DATA_FAILURE 1

/*
 * A star has x, y coordinates and an intensity for each star color.
 */
typedef struct {
    float x, y, intensities[STAR_COLORS];
} star;

typedef struct {
    long min, max;
} minmax;

typedef struct {
    long x_dimension, y_dimension;
} dimensions;

typedef struct {
    dimensions single_panel_pixel_dimensions;
    dimensions panel_indices_dimensions;
    minmax x_pixels;  // Size of a single panel in pixels, for debugging
    minmax y_pixels;  // Size of a single panel in pixels, for debugging
} star_meta_data;

#ifdef __NVCC__
__host__ __device__
#endif
inline int panel_index_lookup(const float x, const float y, const star_meta_data metadata) {
	const float panel_x = x / metadata.single_panel_pixel_dimensions.x_dimension + 1;
	if( panel_x < 0 || panel_x >= metadata.panel_indices_dimensions.x_dimension) return -1;
	const float panel_y = y / metadata.single_panel_pixel_dimensions.y_dimension + 1;
	if(panel_y < 0 || panel_y >= metadata.panel_indices_dimensions.y_dimension) return -1;
    return ((int) y) * panel_indices_dimensions.x_dimension + ((int) x);
}

/*
 * Star data has an array of stars, organized into groups of panels, and an array of panel indices
 */
typedef struct {
    star *stars;
    /* panel_indices has (panel_indices_dimensions[0] * panel_indices_dimensions[1] + 1) elements
     * the final element is number of elements in the star array in data. */
    long *panel_indices;
    star_meta_data metadata;
} star_data;

int parse_star_data_from_tsv(star_data *data,
                             const char *file_name,
                             const minmax x_pixels,
                             const minmax y_pixels,
                             const long panels_per_side);

void star_data_release(star_data data);

#ifdef __cplusplus
}
#endif

#endif //KABUKINAI_STAR_DATA_H
