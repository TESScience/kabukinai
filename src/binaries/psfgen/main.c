#include <stdio.h>
#include <stdlib.h>
#include "../../libraries/argparse/argparse.h"
#include "../../libraries/psf/psf.h"

static const char *description =
        "\npsfgen is a program for creating a FITS file containing a TESS point spread function image";

static const char *usage[] = {
        "psfgen [options] output.fits",
        NULL,
};

int main(int argc, const char *argv[]) {
    double gamma = 0.5;
    char gamma_description[256];
    snprintf(gamma_description,
             sizeof(gamma_description),
             "the width of the Cauchy point spread function, defaults to %g",
             gamma);

    int side_length = 32;
    char side_length_description[256];
    snprintf(side_length_description,
             sizeof(side_length_description),
             "the length of one side of the square output array in CCD pixels, defaults to %i", side_length);

    int oversampling = 3;
    char oversampling_description[256];
    snprintf(oversampling_description,
             sizeof(oversampling_description),
             "the oversampling of the CCD pixel in the output array, defaults to %i", oversampling);

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

    const char *file_name = argv[0];
    char history[1024];
    snprintf(history, sizeof(history), "psgen --gamma %g --side-length %d --oversampling %d %s",
             gamma,
             side_length,
             oversampling,
             file_name);

    psf_data data;
    if (psf_data_init(&data, gamma, side_length, oversampling) != KABUKINAI_PSF_SUCCESS) {
        fprintf(stderr, "Could not initialize PSF data\n");
        exit(EXIT_FAILURE);
    }
    if (write_psf_fits(data, file_name, history) != KABUKINAI_PSF_SUCCESS) {
        fprintf(stderr, "Could not write FITS file \"%s\"\n", file_name);
        psf_data_release(data);
        exit(EXIT_FAILURE);
    }
    psf_data_release(data);
    exit(EXIT_SUCCESS);
}
