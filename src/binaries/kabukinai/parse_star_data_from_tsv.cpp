#include "star_data.h"
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

std::vector<star> parse_star_vector_from_tsv(star_array *array, const char * file_name) {
	// TODO: Handle failure
	std::ifstream infile(file_name);
	std::vector<star> star_vector;
	std::string line;
        float intensities[8];
	while(std::getline(infile, line)) {
		const char * cline = line.c_str();
		star new_star;
		int parsed_arguments = sscanf(cline, "%g %g %g %g %g %g %g %g %g %g",
                                                     &new_star.x, 
                                                     &new_star.y,
                                                     &intensities[0], &intensities[1], 
						     &intensities[2], &intensities[3],
                                                     &intensities[4], &intensities[5], 
						     &intensities[6], &intensities[7]);
		if(parsed_arguments != STAR_COLORS + 2) {
			char error_message[1024];
			snprintf(error_message, sizeof(error_message),
                                 "Could not parse line of TSV:\n\n %s", cline);
			throw std::length_error(error_message);
		}
		for(int color_index = 0; color_index < STAR_COLORS; ++color_index)
			new_star.intensities[color_index] = intensities[color_index];
		star_vector.push_back(new_star);
	}
	return star_vector;
}

long panel_index_lookup(const long x, const long y, const long panel_indexes_dimensions[2]) {
	return x * panel_indexes_dimensions.y_dimension + y;
}

int star_data_from_vector(star_data * data, 
		          const std::vector<star> stars, 
			  const minmax x_pixels,
			  const minmax y_pixels,
			  const long panels_per_side) {
	// TODO: error on negative
	const long x_dimension = x_pixels.max - x_pixels.min;
	const long y_dimension = y_pixels.max - y_pixels.min;
	// TODO: error if not evenly divisible by panels_per_side
	const long single_panel_pixel_dimensions = {
		x_dimension / panels_per_side,
		y_dimension / panels_per_side,
	};
	const long panel_indexes_dimensions = {panels_per_side, panels_per_side};
	star_data -> metadata.x_pixels = x_pixels;
	star_data -> metadata.y_pixels = y_pixels;
	star_data -> metadata.single_panel_pixel_dimensions = single_panel_pixel_dimensions;
	star_data -> metadata.panel_indexes_dimensions = panel_indexes_dimensions;
	std::vector<star> panel_intermediate_data[panels_per_side * panels_per_side];
	for (auto &some_star : stars) {
		const long bin_x = ((some_star -> x) - x_pixels.min) / panels_per_side;
		const long bin_y = ((some_star -> y) - y_pixels.min) / panels_per_side;
		panel_intermediate_data[panel_index_lookup(bin_x, bin_y, panel_indexes_dimensions)].push_back(*star);
	}
	// TODO: flatten panel_intermediate_data into data -> stars
	// TODO: Add panel indexes to data -> panel_indexes
}

extern "C" {
	int parse_star_data_from_tsv(star_data *data, const char * file_name) {
		return -1;
	}
}
