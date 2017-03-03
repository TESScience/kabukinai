#include "star_data.h"

void star_array_release(star_array array) {
	free(array.stars);
}

void star_data_release(star_data data) {
	free(data.panel_indexes);
	star_array_release(data.array);
}

