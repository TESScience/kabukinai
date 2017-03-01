#ifndef KABUKINAI_PSF_H
#define KABUKINAI_PSF_H

double cauchy_psf(double x, double y, double g);

#endif //KABUKINAI_PSF_H
int write_psf_fits(double gamma, int side_length, int oversampling, const char* fits_file_name);
