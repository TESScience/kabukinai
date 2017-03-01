#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../../libraries/cfitsio/fitsio.h"
#include "psf.h"

#define SUCCESS EXIT_SUCCESS
#define FAILURE EXIT_FAILURE

#define PANIC_ON_BAD_FITSIO_STATUS(error_message, status) { \
	const int status_value = (status); \
	if (status_value != 0) { \
		fprintf(stderr, "FAILURE: %s", (error_message)); \
		fits_report_error(stderr, status_value); \
		return FAILURE; \
	} \
}

int write_psf_fits(double gamma, int side_length, int oversampling, const char* fits_file_name) {
	if(access(fits_file_name, F_OK) != -1 && remove(fits_file_name) != 0) {
		perror("Output file cannot be written");
		return FAILURE;
	}

	int status = 0;
	fitsfile *fits_file_pointer;
	fits_create_file(&fits_file_pointer, fits_file_name, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not create file", status);

	long dimensions[2];
	dimensions[0] = dimensions[1] = side_length * oversampling;
	fits_create_img(fits_file_pointer, FLOAT_IMG, 2, dimensions, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not create image", status);

	const double mid_column = (dimensions[0] - 1.0) / 2.0;
	const double mid_row = (dimensions[1] - 1.0) / 2.0;

	float * image_pixels = calloc(dimensions[0]*dimensions[1], sizeof(float));
	for(int column = 0; column < dimensions[0]; ++column)
		for(int row = 0; row < dimensions[1]; ++row)
			image_pixels[row * dimensions[0] + column] = cauchy_psf((column - mid_column) / oversampling, 
					                                        (row - mid_row) / oversampling, 
										gamma);
	long fpixel[2] = {1,1};
	fits_write_pix(fits_file_pointer, TFLOAT, fpixel, dimensions[0] * dimensions[1], image_pixels, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not write image", status);

	fits_close_file(fits_file_pointer, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not close file", status);
	return SUCCESS;
}
