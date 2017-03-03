#include "star_data.h"
#include <malloc.h>


void star_data_release(star_data data) {
	free(data.panel_indexes);
	free(data.stars);
}

