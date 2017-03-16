#ifndef KABUKINAI_PSF_H
#define KABUKINAI_PSF_H

#define KABUKINAI_PSF_SUCCESS 0
#define KABUKINAI_PSF_FAILURE 1

#ifdef __cplusplus
extern "C" {
#endif

double cauchy_psf(double x, double y, double g);

typedef struct {
	double gamma;
	int side_length;
	int oversampling;
	long dimensions[2];
	float * image_pixels;
} psf_data;

int psf_data_init(psf_data * data, double gamma, int side_length, int oversampling);
int write_psf_fits(psf_data data, const char * fits_file_name, const char * history);
int read_psf_fits(psf_data * data, const char * fits_file_name);
void psf_data_release(psf_data data);

#ifdef __cplusplus
}
#endif

#endif //KABUKINAI_PSF_H
