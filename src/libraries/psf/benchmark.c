#include <stdio.h>
#include <stdlib.h>
#include "psf.h"
#include <time.h>

int main() {
    const double start_time = (double) clock() / CLOCKS_PER_SEC;
    double total = 0;
    srand(0);
    for (int i = 0; i < 1000000; ++i)
        total += cauchy_psf(10 * rand() - 50, 10 * rand() - 5, 10 * rand());
    printf("total: %g\n", total);
    const double end_time = (double) clock() / CLOCKS_PER_SEC;
    printf("time: %g seconds\n", end_time - start_time);
    exit(EXIT_SUCCESS);
}
