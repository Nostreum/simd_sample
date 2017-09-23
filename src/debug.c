#include "hvadd.h"

void print128(float128 v){

	f32 *vf32 = (f32*)&v;
	printf("[ %.3f, %.3f, %.3f, %.3f ]", v[3], v[2], v[1], v[0]);

}

void print256(float256 v){

	f32 *vf32 = (f32*)&v;
	printf("[ %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f ]", v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]);

}