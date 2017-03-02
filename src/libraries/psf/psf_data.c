#include <stdio.h>
#include <stdlib.h>
#include "psf.h"

int psf_data_init(psf_data * data, const double gamma, const int side_length, const int oversampling) {
	data -> gamma = gamma;
	data -> side_length = side_length;
	data -> oversampling = oversampling;
	data -> dimensions[0] = data -> dimensions[1] = side_length * oversampling;
	data -> image_pixels = calloc((data -> dimensions[0]) * (data -> dimensions[1]), sizeof(float));
	if ((data -> image_pixels) == NULL) {
		fprintf(stderr, "Could not allocate memory for image pixels\n");
		return KABUKINAI_PSF_FAILURE;
	}
	const double mid_column = (data -> dimensions[0] - 1.0) / 2.0;
	const double mid_row = (data -> dimensions[1] - 1.0) / 2.0;
	for(int column = 0; column < data -> dimensions[0]; ++column)
		for(int row = 0; row < data -> dimensions[1]; ++row)
			data -> image_pixels[row * data -> dimensions[0] + column] = cauchy_psf((column - mid_column) / oversampling, 
					                                                        (row - mid_row) / oversampling, 
                                                                                                gamma);
	return KABUKINAI_PSF_SUCCESS;
}

void psf_data_release(psf_data data) {
	free(data.image_pixels);
}
