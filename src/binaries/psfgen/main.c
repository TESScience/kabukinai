#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../../libraries/argparse/argparse.h"
#include "../../libraries/cfitsio/fitsio.h"
#include "psf.h"

static const char* description = "\npsfgen is a program for creating a FITS file containing a TESS point spread function image";

static const char* usage[] = {
	"psfgen [options] output.fits",
	NULL,
};

void panic_on_bad_status(const char * error_message, int status) {
	if (status != 0) {
		fprintf(stderr, "FAILURE: %s", error_message);
		fits_report_error(stderr, status);
		exit(EXIT_FAILURE);
	}
}

int main(const int argc, const char* argv[]) {
	double gamma = 0.5;
	char gamma_description[256];
	snprintf(gamma_description, 256, "the width of the Cauchy point spread function, defaults to %g", gamma);

	int side_length = 32;
	char side_length_description[256];
	snprintf(side_length_description, 256, "the length of one side of the square output array in CCD pixels, defaults to %i", side_length);

	int oversampling = 3; 
	char oversampling_description[256];
	snprintf(side_length_description, 256,"the oversampling of the CCD pixel in the output array, defaults to %i", oversampling);

	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_DOUBLE('g', "gamma", &gamma, gamma_description),
		OPT_INTEGER('s', "side-length", &side_length, side_length_description),
		OPT_INTEGER('o', "oversampling", &oversampling, oversampling_description),
		OPT_END(),
	};

	struct argparse parser;
	argparse_init(&parser, options, usage, 0);
	argparse_describe(&parser, description, "");
	const int unread_arguments = argparse_parse(&parser, argc, argv);

	if (unread_arguments > 1) {
		fprintf(stderr, "More than one output file given; exactly one must be specified\n");
		exit(EXIT_FAILURE);
	}

	if (unread_arguments == 0) {
		fprintf(stderr, "No output file specified\n");
		exit(EXIT_FAILURE);
	}

	const char * fits_file_name = argv[0];
	if(access(fits_file_name, F_OK) != -1) {
		const int status = remove(fits_file_name);
		if (status != 0) {
			perror("Output file cannot be written");
			exit(EXIT_FAILURE);
		}
	}

	int status = 0;
	fitsfile *fits_file_pointer;
	fits_create_file(&fits_file_pointer, fits_file_name, &status);
	panic_on_bad_status("Could not create file", status);

	long dimensions[2];
	dimensions[0] = dimensions[1] = side_length * oversampling;
	fits_create_img(fits_file_pointer, FLOAT_IMG, 2, dimensions, &status);
	panic_on_bad_status("Could not create image", status);

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
	panic_on_bad_status("Could not write image", status);

	fits_close_file(fits_file_pointer, &status);
	panic_on_bad_status("Could not close file", status);

	exit(EXIT_SUCCESS);
}
