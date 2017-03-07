#include <stdio.h>

/*
 * TODO make comment regarding matrix orientation versus image coordinates.
 */

/*
 * Texture reference. File scope, mutable, shared between device
 * and host. Holds details of the texture. The stuff in <> is immutable.
 * Other details are mutable, and are set in host code, including
 * its binding to a data array.
 */

texture<float, cudaTextureType2D, cudaReadModeElementType> myTexture;

__global__ void try_interpolation( float *output ) {

/*
 * Distort our image to test interpolation and clipping.
 */
	
	float y = threadIdx.y * 0.75 - 0.5;
	float x = threadIdx.x *0.5;
	
	output[ threadIdx.y * blockDim.x + threadIdx.x ] =
		tex2D( myTexture, y, x );
}

int main()
{
	cudaError_t code;

/*
 * Make a 2d test pattern as an ordinary C array.
 */
 
	const int width = 4 , height = 4 ;

	float data[height][width] ;
	for ( int y =0 ; y<height ; y++ ) {
			for ( int x = 0 ; x<width; x++ ){
			data[y][x] = x*x+y*y;
		}
	}
	const int size = width*height*sizeof(float) ;

/*
 * Print the contents of the input array.
 */	
	
	for ( int y =0 ; y<height ; y++ ) {
			for ( int x = 0 ; x<width; x++ ){
			printf( "%10g ", data[y][x]);
		}
		printf( "\n");
	}
	printf( "\n" );

/*
 * A cudaChannelFormatDesc is a structure that defines the
 * contents of an element of a cudaArray. Such an element can contain
 * up to four numbers. cudaCreateChannelDesc() allows you to specify the 
 * length of each of the four in bits (!) along with a type category
 * common to all. So, this complicated call defines an ordinaly scalar
 * float as an array element.
 */

/*
 * A cudaChannelFormatDesc is a structure that defines the
 * contents of an element of a cudaArray.
 * 
 * An element of a cudaArray has up to four numbers of variable length.
 * 
 * struct cudaChannelFormatDesc {
 *    int x, y, z, w;
 *    enum cudaChannelFormatKind  f;
 * }

It is constructed with a function `cudaCreateChannelDesc`
cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind  f) ;

Here x,y,z, and w are the number of bits for each dimension
*/
 
	const cudaChannelFormatDesc floatDesc = 
		cudaCreateChannelDesc(32, 0, 0, 0,
		cudaChannelFormatKindFloat );

/*
 * A cudaArray is an array of objects defined by a cudaChannelFormatDesc.
 * Allocate one of these on the device.
 */
 
	cudaArray* floatArray;
	code = cudaMallocArray(&floatArray, &floatDesc, height, width);
	if( code ) {
		printf( "cudaMallocArray: %s\n",
			cudaGetErrorString(code));
		exit( 1 );
	}

/*
 * Now copy our ordinary C array to the cudaArray on the device.
 */

	code = cudaMemcpyToArray(floatArray, 0, 0, data, size,
		cudaMemcpyHostToDevice);
	if( code ) {
		printf( "cudaMemcpyToArray: %s\n",
			cudaGetErrorString(code));
		exit( 1 );
	}


/*
 * Now set up the mutable fields of the texture reference.
 *
 * First, set it up to yield zero for requests outside the array.
 */
 
	myTexture.addressMode[0] = cudaAddressModeBorder;
	myTexture.addressMode[1] = cudaAddressModeBorder;

/*
 * Interpolate between samples.
 */
	
	myTexture.filterMode = cudaFilterModeLinear;

/*
 * Use [0,height-1][0, width-1] as the range for the floating 
 * coordinates.
 */
	
	myTexture.normalized = false;

/*
 * Finally, bind the texture to its data.
 */
	
	code = cudaBindTextureToArray(myTexture, floatArray, floatDesc);
	if( code ) {
		printf( "cudaBindTextureToArray: %s\n",
			cudaGetErrorString(code));
		exit( 1 );
	}

/*
 * For no good reason, make the output array
 * the same size as the texture.
 */
	
	float* output;
	code =cudaMalloc(&output, size );
	if( code ) {
		printf( "cudaMalloc: %s\n", cudaGetErrorString(code));
		exit( 1 );
	}
/*
 * Run the kernel. Note that we don't have to tell it sizes
 * of things in the args, as those are implied by the block dimensions.
 */
	dim3 blocks_dimension( height, width );
	try_interpolation<<< 1, blocks_dimension  >>>
		( output );
	code = cudaDeviceSynchronize();
	if( code ) {
		printf( "cudaDeviceSynchronize: %s\n",
			cudaGetErrorString(code));
		exit( 1 );
	}

/*
 * Copy the result back to the host.
 */

	float result[height][width];
	code = cudaMemcpy(result, output, size, cudaMemcpyDeviceToHost);
	if( code ){
		printf( "cudaMemcpyDeviceToHost: %s\n",
			cudaGetErrorString(code));
		exit( 1 );
	}
/*
 * Print the result.
 */
	
	for ( int y =0 ; y<height ; y++ ) {
			for ( int x = 0 ; x<width; x++ ){
			printf( "%10g ", result[y][x]);
		}
		printf( "\n");
	}

/*
 * If you're really done, you can tidy up with a bulldozer ;-)
 */
	
	code = cudaDeviceReset();
	if( code ) {
		printf( "cudaMemcpyDeviceToHost: %s\n",
			cudaGetErrorString(code));
		exit( 1 );
	}
}
