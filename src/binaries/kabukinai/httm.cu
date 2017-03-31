#include "kabukinai.h"
#include "curand_kernel.h"
#include "math.h"

// A kernel to extract slices in reverse, 'cause there's not a cudaMemcpy variant for that
// Launch with a grid dimension of the image height, block dimension of the slice image width

__global__ void extract_reverse_floating_slice(
	float *raster, 
	float *slice, 
	int slice_width, 
	int image_width 
) 
{
	slice[slice_width * blockIdx.x + blockDim.x - 1 - threadIdx.x] = 
		raster[image_width * blockIdx.x + threadIdx.x];
}

// Replace a floating point simulated image with a sliced version
__host__ void
to_slices( simulation_data *d ) {
	float * slices;
	int height = d->dimensions[0];
	int width = d->dimensions[1];
	int slice_image_width = width/d->number_of_slices;
	int slice_width = slice_image_width + d->early_dark_pixels + d->late_dark_pixels;
	int slice_height = height + d->smear_rows + d->final_dark_rows;
	int slice_size = slice_width*slice_height;
	
	PANIC_ON_BAD_CUDA_STATUS(cudaMalloc((void **) &slices, slice_size*d->number_of_slices*sizeof(float)));
		
	for( int s = 0; s < d->number_of_slices; s += 1) {
		float *this_slice = slices + s * slice_size;
		PANIC_ON_BAD_CUDA_STATUS(
			cudaMemset2DAsync( // zero early darks
				this_slice, 
				slice_width * sizeof(float), 
				0, 
				d->early_dark_pixels * sizeof(float), 
				slice_height ));
		PANIC_ON_BAD_CUDA_STATUS(
			cudaMemset2DAsync( // zero late darks
				this_slice + d->early_dark_pixels + slice_image_width, 
				slice_width * sizeof(float), 
				0, 
				d->late_dark_pixels * sizeof(float), 
				slice_height ));
		PANIC_ON_BAD_CUDA_STATUS(
			cudaMemset2DAsync( // sero smear and final darks
				this_slice + height * slice_width, 
				slice_width * sizeof(float), 
				0, 
				slice_width * sizeof(float), 
				d->smear_rows + d->final_dark_rows ));
		
		if( s & 1 ) {	// flip odd slices
			extract_reverse_floating_slice<<<height, slice_image_width>>>( 
				d->image_pixels + s * slice_image_width, 
				this_slice + d->early_dark_pixels,
				slice_width, width );
		} else {
			PANIC_ON_BAD_CUDA_STATUS(		
				cudaMemcpy2DAsync( 
					this_slice + d->early_dark_pixels,
					slice_width * sizeof(float),
					d->image_pixels + s * slice_image_width,
					width * sizeof(float),
					slice_image_width * sizeof(float),
					height,
					cudaMemcpyDeviceToDevice
			));			
		}
	}
	
	PANIC_ON_BAD_CUDA_STATUS(cudaDeviceSynchronize());
	PANIC_ON_BAD_CUDA_STATUS(cudaFree(d->image_pixels));
	d->image_pixels = slices;
}



__global__ void cu_smear( 
	float * slices, 
	int slice_size, 
	int early_darks, 
	int smear_rows, 
	int image_height, 
	int slice_width,
	float smear_ratio
)
{
	double smear = 0.0;
	float *image_pixel = slices + early_darks + blockIdx.x * slice_size + threadIdx.x;
	for( int i = 0; i < image_height; i += 1) {
		smear += *image_pixel;
		image_pixel += slice_width;
	}
	// Note that image_pixel automagically winds up pointing to the first smear row
	
	smear *= smear_ratio;
	
	for( int i = 0; i < smear_rows; i += 1 ) {
		*image_pixel = smear;
		image_pixel += slice_width;
	}
}


__host__ void add_smear( simulation_data *d ) {
	int height = d->dimensions[0];
	int width = d->dimensions[1];
	int slice_image_width = width/d->number_of_slices;
	int slice_width = slice_image_width + d->early_dark_pixels + d->late_dark_pixels;
	int slice_height = height + d->smear_rows + d->final_dark_rows;
	int slice_size = slice_width*slice_height;
	
	cu_smear<<<d->number_of_slices, slice_image_width>>>( 
		d->image_pixels, 
		slice_size, 
		d->early_dark_pixels,
		d->smear_rows,
		height,
		slice_width,
		d->smear_ratio );
		
	PANIC_ON_BAD_CUDA_STATUS(cudaDeviceSynchronize());
}


__global__ void cu_noise( 
	float * slice, 
	int slice_size, 
	int height, 
	int slice_width,
	unsigned long long random_seed,
	unsigned long long random_offset,
	float readout_noise_variance
)
{
	curandState_t random_state;
	curand_init ( 
		random_seed, 
		(unsigned long long) threadIdx.x, 
		random_offset, 
		&random_state );

	float *image_pixel = slice + threadIdx.x;
	for( int i = 0; i < height; i += 1) {
		
		*image_pixel += 
			sqrtf(readout_noise_variance + *image_pixel) * 
			curand_normal (&random_state);
		image_pixel += slice_width;
	}
}


__host__ void add_noise( simulation_data *d ) {
	int height = d->dimensions[0];
	int width = d->dimensions[1];
	int slice_image_width = width/d->number_of_slices;
	int slice_width = slice_image_width + d->early_dark_pixels + d->late_dark_pixels;
	int slice_height = height + d->smear_rows + d->final_dark_rows;
	int slice_size = slice_width*slice_height;
	
	for( int s = 0; s < d->number_of_slices; s += 1 ) {
	
		cu_noise<<<1, slice_image_width>>>( 
			d->image_pixels + s * slice_size, 
			slice_size, 
			height,
			slice_width,
			d->random_seed,
			d->random_offset + s * 1000000,
			d->read_noise_variance[s]
		);
	}
		
	PANIC_ON_BAD_CUDA_STATUS(cudaDeviceSynchronize());
	
	d->random_offset += 10000000;	// be sure we get new numbers next call
}

