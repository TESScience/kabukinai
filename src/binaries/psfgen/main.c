#include <stdio.h>
#include <stdlib.h>
#include "../../libraries/argparse/argparse.h"
#include "psf.h"

static const char* description = "\npsfgen is a program for creating a FITS file containing a TESS point spread function image";

static const char* usage[] = {
	"psfgen [options] output.fits",
	NULL,
};

int main(int argc, const char* argv[]) {
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
	argc = argparse_parse(&parser, argc, argv);

	if (argc > 1) {
		fprintf(stderr, "More than one output file given; exactly one must be specified\n");
		exit(EXIT_FAILURE);
	}

	if (argc == 0) {
		fprintf(stderr, "No output file specified\n");
		exit(EXIT_FAILURE);
	}

	exit(write_psf_fits(gamma, side_length, oversampling, argv[0]));
}
