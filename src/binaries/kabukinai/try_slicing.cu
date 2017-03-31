#include "kabukinai.h"
#include <stdio.h>


const float expected[] = 
{ 0.0, 0.0, 0.0, 1.0, 2.0, 0.0 ,
0.0, 0.0,12.0,13.0,14.0, 0.0 ,
0.0, 0.0,24.0,25.0,26.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 5.0, 4.0, 3.0, 0.0 ,
0.0, 0.0,17.0,16.0,15.0, 0.0 ,
0.0, 0.0,29.0,28.0,27.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 6.0, 7.0, 8.0, 0.0 ,
0.0, 0.0,18.0,19.0,20.0, 0.0 ,
0.0, 0.0,30.0,31.0,32.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0,11.0,10.0, 9.0, 0.0 ,
0.0, 0.0,23.0,22.0,21.0, 0.0 ,
0.0, 0.0,35.0,34.0,33.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

int main() {
	simulation_data d;
	float image_pixels[4*8*6];	// enough to hold sliced image
	
	
	for( int i = 0; i < 36; i += 1 ){
		image_pixels[i] = (float) i;
	}
	
	PANIC_ON_BAD_CUDA_STATUS(cudaMalloc((void **) &d.image_pixels, 36*sizeof(float)));
	PANIC_ON_BAD_CUDA_STATUS(cudaMemcpy(d.image_pixels, image_pixels,
		36*sizeof(float),cudaMemcpyHostToDevice));
	d.dimensions[0] = 3;
	d.dimensions[1] = 12;
	d.number_of_slices = 4;
	d.early_dark_pixels = 2;
	d.late_dark_pixels = 1;
	d.smear_rows = 3;
	d.final_dark_rows = 2;
	
	printf( "Calling to_slices: " );
	to_slices( &d );
	printf( "success!\n" );
	
	
	PANIC_ON_BAD_CUDA_STATUS(cudaMemcpy(image_pixels, d.image_pixels,
		4*8*6*sizeof(float),cudaMemcpyDeviceToHost));
	
	int result = KABUKINAI_SUCCESS;

	for( int i = 0; i < 4*8*6; i+=1 ) {
		if( image_pixels[i] != expected[i] ) {
			printf( "At pixel %d expected %f got %f\n", i, expected[i], image_pixels[i] );
			result = KABUKINAI_FAILURE;
		}
	}
	
	return result;
}
