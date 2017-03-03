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
	float x, y, intensities[ STAR_COLORS ];
} star;

typedef struct {
	long min, max;
} minmax;

typedef struct {
	long x_dimension, y_dimension;
} dimensions;

typedef star_metadata {
	dimensions single_panel_pixel_dimensions; // Size of a single panel in pixels, for debugging
	dimensions panel_indexes_dimensions;
	minmax x_pixels;
	minmax y_pixels;
} star_meta_data;

/*
 * Star data has an array of stars, organized into groups of panels, and an array of panel indexes
 */
typedef struct {
	star * stars;
	/* panel_indexes has (panel_indexes_dimensions[0] * panel_indexes_dimensions[1] + 1) elements 
	 * the final element is number of elements in the star array in data. */
	long * panel_indexes; 
	star_meta_data metadata;
} star_data;

int parse_star_data_from_tsv(star_data *data, const char * file_name);
void star_data_release(star_data data);

#ifdef __cplusplus
}
#endif

#endif //KABUKINAI_STAR_DATA_H
