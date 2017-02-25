#include <math.h>

double cauchy_psf(double x, double y, double g) {
    const double twice_g = g + g;
    const double twice_x = x + x;
    const double twice_y = y + y;

    const double g2 = g * g;
    const double y2 = y * y;
    const double x2 = x * x;

    const double y2py = y2 + y;
    const double x2px = x2 + x;
    const double y2my = y2 - y;
    const double x2mx = x2 - x;

    const double a = g2 + x2mx;
    const double b = g2 + x2px;

    const double t0 = 2 + 4 * (a + y2my);
    const double rt0 = sqrt(t0);
    const double t1 = 2 + 4 * (a + y2py);
    const double rt1 = sqrt(t1);
    const double t2 = 2 + 4 * (b + y2my);
    const double rt2 = sqrt(t2);
    const double t3 = 2 + 4 * (b + y2py);
    const double rt3 = sqrt(t3);

    const double cx0 = twice_x - 1;
    const double cx1 = twice_x + 1;
    const double cy0 = twice_y - 1;
    const double cy1 = twice_y + 1;

    return (atan2(twice_g * (t0 + cy0 * rt0), -cx0 * cy0 * (cy0 + rt0)) +
            atan2(twice_g * (t1 + cy1 * rt1), cx0 * cy1 * (cy1 + rt1)) +
            atan2(-twice_g * (t2 + cy0 * rt2), -cx1 * cy0 * (cy0 + rt2)) +
            atan2(-twice_g * (t3 + cy1 * rt3), cx1 * cy1 * (cy1 + rt3))) * .5 * M_1_PI;
}

