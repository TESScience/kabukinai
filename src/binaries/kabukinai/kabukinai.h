#ifndef KABUKINAI_H
#define KABUKINAI_H

#define KABUKINAI_SUCCESS 0
#define KABUKINAI_FAILURE 1

typedef struct {
	float * image_pixels;
	long dimensions[2];
} simulation_data;

int write_simulation_fits(simulation_data data, const char * fits_file_name, const char * history);

#endif //KABUKINAI_H
