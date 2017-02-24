#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "psf.h"

int main(int argc, const char * argv[]) {
    assert(cauchy_psf(1,1,1) - 0.0323865 < 0.000001);
    exit(EXIT_SUCCESS);
}
