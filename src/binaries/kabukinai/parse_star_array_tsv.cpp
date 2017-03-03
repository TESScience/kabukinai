#include "star_data.h"
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
	int parse_star_array_tsv(star_array *array, const char * file_name) {
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
				fprintf(stderr, "Could not parse line of TSV:\n\n %s", cline);
				return KABUKINAI_STAR_DATA_FAILURE;
			}
			for(int color_index = 0; color_index < STAR_COLORS; ++color_index)
				new_star.intensities[color_index] = intensities[color_index];
			star_vector.push_back(new_star);
		}
		array -> stars = (star*) calloc(star_vector.size(), sizeof(star));
		std::copy(star_vector.begin(), star_vector.end(), array -> stars);
		array -> count = star_vector.size();
		return KABUKINAI_STAR_DATA_SUCCESS;
	}
}
