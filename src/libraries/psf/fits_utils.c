#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../cfitsio/fitsio.h"
#include "psf.h"

#define PANIC_ON_BAD_FITSIO_STATUS(error_message, status) { \
	const int status_value = (status); \
	if (status_value != 0) { \
		fprintf(stderr, "KABUKINAI_PSF_FAILURE: %s at line %d in file %s\n", (error_message), __LINE__, __FILE__); \
		fits_report_error(stderr, status_value); \
		return KABUKINAI_PSF_FAILURE; \
	} \
}

int write_psf_fits(psf_data data, const char* fits_file_name, const char* history) {
	if(access(fits_file_name, F_OK) != -1 && remove(fits_file_name) != 0) {
		perror("Output file cannot be written");
		return KABUKINAI_PSF_FAILURE;
	}

	int status = 0;

	fitsfile *fits_file_pointer;
	fits_create_file(&fits_file_pointer, fits_file_name, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not create file", status);

	fits_create_img(fits_file_pointer, FLOAT_IMG, 2, data.dimensions, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not create image", status);

	fits_write_key(fits_file_pointer, TDOUBLE, "GAMMA", &(data.gamma), "Width of Cauchy distribution", &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not write GAMMA keyword in header", status);

	fits_write_key(fits_file_pointer, TINT, "SLENGTH", &(data.side_length), "Length of image side in pixels", &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not write SLENGTH keyword in header", status);

	fits_write_key(fits_file_pointer, TINT, "OVRSMPL", &(data.oversampling), "Image pixels per CCD pixel", &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not write OVRSMPL keyword in header", status);

	fits_write_history(fits_file_pointer, history, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not append to HISTORY keyword in header", status);

	long fpixel[2] = {1,1};
	LONGLONG image_pixels_size = data.dimensions[0] * data.dimensions[1];
	fits_write_pix(fits_file_pointer, TFLOAT, fpixel, image_pixels_size, data.image_pixels, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not write image pixels", status);

	fits_close_file(fits_file_pointer, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not close file", status);
	return KABUKINAI_PSF_SUCCESS;
}

int read_psf_fits(psf_data * data, const char* fits_file_name) {
	int status = 0;

	fitsfile *fits_file_pointer;
	fits_open_image(&fits_file_pointer,  fits_file_name, READONLY, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not open file", status);

	fits_read_key(fits_file_pointer, TDOUBLE, "GAMMA", &(data -> gamma), NULL, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not read GAMMA keyword in header", status);

	fits_read_key(fits_file_pointer, TINT, "SLENGTH", &(data -> side_length), NULL, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not read SLENGTH keyword in header", status);

	fits_read_key(fits_file_pointer, TINT, "OVRSMPL", &(data -> oversampling), NULL, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not read OVRSMPL keyword in header", status);

	fits_read_key(fits_file_pointer, TLONG, "NAXIS1", &(data -> dimensions[0]), NULL, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not read NAXIS1 keyword in header", status);

	fits_read_key(fits_file_pointer, TLONG, "NAXIS2", &(data -> dimensions[1]), NULL, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not read NAXIS2 keyword in header", status);

	LONGLONG image_pixels_size = data -> dimensions[0] * data -> dimensions[1];

	data -> image_pixels = (float*) calloc(image_pixels_size, sizeof(float));

	if ((data -> image_pixels) == NULL) {
		fprintf(stderr, "Could not allocate memory for image pixels\n");
		return KABUKINAI_PSF_FAILURE;
	}


	long fpixel[2] = {1,1};
	float null_value = -1;
	int any_null = 0;
	fits_read_pix(fits_file_pointer, TFLOAT, fpixel, image_pixels_size, &null_value, data -> image_pixels, &any_null, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not read image pixels", status);

	fits_close_file(fits_file_pointer, &status);
	PANIC_ON_BAD_FITSIO_STATUS("Could not close file", status);
	return KABUKINAI_PSF_SUCCESS;
}
