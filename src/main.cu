#include <iostream>
#include <stdlib.h>
#include "externalClass.cu" // important to include .cu file, not header file
#include "libraries/cfitsio/fitsio.h"

int main(int argc, char *argv[]) {
	externalClass myStuff;

	std::cout << "This is just a plain, host-generated 5, isn't it?: " << myStuff.GetInt() << std::endl;

	const int localN = 10;
	double * localFloat = new double[localN];
	for (int i = 0; i < localN; i++)
		localFloat[i] = i;

	myStuff.squareOnDevice(localFloat, localN);
	
	std::cout << "Final squared values are: " << std::endl;
	
	for (int i = 0; i < localN; i++)
		std::cout << localFloat[i] << std::endl;

	fitsfile *fptr;
	int status = 0;
	fits_open_file(&fptr, argv[1], READONLY, &status);
	if (status == 1) {
		std::cout << "Program read fits input cleanly" << std::endl;
		return EXIT_SUCCESS;
	} else {
		std::cerr << "Program failed to read input" << std::endl;
		return EXIT_FAILURE;
	}
}
