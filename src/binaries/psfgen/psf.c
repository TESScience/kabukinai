#include <math.h>

#define ARCTAN(x, y) atan2((y),(x))

double cauchy_psf(double x, double y, double g) {
    return (ARCTAN(-((-1 + 2 * x) * (-1 + 2 * y) *
                     (-1 + 2 * y + sqrt(2 + 4 * g * g + 4 * (-1 + x) * x + 4 * (-1 + y) * y))),
                   -2 * g * (-2 - 4 * g * g + 4 * x - 4 * x * x + 4 * y - 4 * y * y +
                             (1 - 2 * y) * sqrt(2 + 4 * g * g + 4 * (-1 + x) * x + 4 * (-1 + y) * y))) +
            ARCTAN(-((1 + 2 * x) * (-1 + 2 * y) *
                     (-1 + 2 * y + sqrt(2 + 4 * g * g + 4 * x * (1 + x) + 4 * (-1 + y) * y))),
                   -2 * g * (2 + 4 * g * g + 4 * x + 4 * x * x - 4 * y + 4 * y * y +
                             (-1 + 2 * y) * sqrt(2 + 4 * g * g + 4 * x * (1 + x) + 4 * (-1 + y) * y))) +
            ARCTAN((-1 + 2 * x) * (1 + 2 * y) * (1 + 2 * y + sqrt(2 + 4 * g * g + 4 * (-1 + x) * x + 4 * y * (1 + y))),
                   -2 * g * (-2 * (1 + 2 * g * g + 2 * (-1 + x) * x + 2 * y * (1 + y)) -
                             (1 + 2 * y) * sqrt(2 + 4 * g * g + 4 * (-1 + x) * x + 4 * y * (1 + y)))) +
            ARCTAN((1 + 2 * x) * (1 + 2 * y) * (1 + 2 * y + sqrt(2 + 4 * g * g + 4 * x * (1 + x) + 4 * y * (1 + y))),
                   -2 * g * (2 + 4 * g * g + 4 * x + 4 * x * x + 4 * y + 4 * y * y +
                             (1 + 2 * y) * sqrt(2 + 4 * g * g + 4 * x * (1 + x) + 4 * y * (1 + y))))) * .5 * M_1_PI;
}

