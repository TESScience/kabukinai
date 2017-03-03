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
	long count;
	star * stars;
} star_array;

int parse_star_array_tsv(star_array *stars, const char * file_name);
void star_array_release(star_array *stars);

/*
 * Star data has an array of stars, organized into groups of panels, and an array of panel indexes
 */
typedef struct {
	star_array array;
	/* panel_indexes has (panel_indexes_dimensions[0] * panel_indexes_dimensions[1] + 1) elements 
	 * the final element is number of elements in the star array in data. */
	long * panel_indexes; 
	long single_panel_pixel_dimensions[2]; // Size of a single panel in pixels, for debugging
	long panel_indexes_dimensions[2];
} star_data;

int star_data_init(star_data * star_data, star_array star_array);
void star_data_release(star_data data);

#ifdef __cplusplus
}
#endif

#endif //KABUKINAI_STAR_DATA_H
