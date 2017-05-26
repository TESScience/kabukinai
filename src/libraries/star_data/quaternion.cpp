#include "star_data.h"
#include <math.h>

point_3d sky_coordinates_to_3d(const sky_coordinate coordinate) {
	return {cos(coordinate.dec) * sin(coordinate.dec),
	       	sin(coordinate.dec), 
		cos(coordinate.dec) * cos(coordinate.dec)};
}

point_3d quaternion_rotate(const quaternion q, const point_3d p) {
	return {q.w*q.w*p.x + q.i*q.i*p.x - (q.j*q.j + q.k*q.k)*p.x + q.w*(-2*q.k*p.y + 2*q.j*p.z) + 2*q.i*(q.j*p.y + q.k*p.z),
		2*q.i*q.j*p.x + 2*q.w*q.k*p.x + q.w*q.w*p.y - q.i*q.i*p.y + q.j*q.j*p.y - q.k*q.k*p.y - 2*q.w*q.i*p.z + 2*q.j*q.k*p.z,
		-2*q.w*q.j*p.x + 2*q.i*q.k*p.x + 2*q.w*q.i*p.y + 2*q.j*q.k*p.y + q.w*q.w*p.z - q.i*q.i*p.z - q.j*q.j*p.z + q.k*q.k*p.z};
}

point_2d gnomonic_projection(const point_3d p) {
	return {(float) (p.x / p.z), (float) (p.y / p.z)};
}

point_2d apply_jitter(const quaternion q, const point_2d p) {
	return {(float) ((2*q.j + p.x - 2*q.k*p.y)/(1 - 2*q.j*p.x + 2*q.i*p.y)), 
		(float) ((-2*q.i + 2*q.k*p.x + p.y)/(1 - 2*q.j*p.x + 2*q.i*p.y))};
}	

// To test: Make a rotation matrix in Mathematica, have it act on a 3-vector and compute a corresponding quaternion.  
//          The rotation action of the quaternion in C should agree with the matrix rotation.
//
//          Make a random infinitesimal rotation q (where w is 1 and i,j,k are < 10E-6) and verify for random 3D 
//          point p and check that gnomonic_projection(quaternion_rotate(q,p)) ~ apply_jitter(q, gnomonic_projection(p))
