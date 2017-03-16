#include "../../libraries/cfitsio/fitsio.h"
#include "kabukinai.h"
#include <stdio.h>
#include <unistd.h>

#define PANIC_ON_BAD_FITSIO_STATUS(error_message, status) { \
        const int status_value = (status); \
        if (status_value != 0) { \
                fprintf(stderr, "KABUKINAI_FAILURE: %s", (error_message)); \
                fits_report_error(stderr, status_value); \
                return KABUKINAI_FAILURE; \
        } \
}

int write_simulation_fits(simulation_data data, const char* fits_file_name, const char* history) {
        if(access(fits_file_name, F_OK) != -1 && remove(fits_file_name) != 0) {
                perror("Output file cannot be written");
                return KABUKINAI_FAILURE;
        }

        int status = 0;

        fitsfile *fits_file_pointer;
        fits_create_file(&fits_file_pointer, fits_file_name, &status);
        PANIC_ON_BAD_FITSIO_STATUS("Could not create file", status);

        fits_create_img(fits_file_pointer, FLOAT_IMG, 2, data.dimensions, &status);
        PANIC_ON_BAD_FITSIO_STATUS("Could not create image", status);

        fits_write_history(fits_file_pointer, history, &status);
        PANIC_ON_BAD_FITSIO_STATUS("Could not append to HISTORY keyword in header", status);

        long fpixel[2] = {1,1};
        LONGLONG image_pixels_size = data.dimensions[0] * data.dimensions[1];
        fits_write_pix(fits_file_pointer, TFLOAT, fpixel, image_pixels_size, data.image_pixels, &status);
        PANIC_ON_BAD_FITSIO_STATUS("Could not write image pixels", status);

        fits_close_file(fits_file_pointer, &status);
        PANIC_ON_BAD_FITSIO_STATUS("Could not close file", status);
        return KABUKINAI_SUCCESS;
}
