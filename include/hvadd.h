#ifndef _HVADD_H
#define _HVADD_H

#include <stdio.h>

#include <nmmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

#include <x86intrin.h>

#define SIMD_ALIGNMENT 4
#define AVX_ALIGNMENT 32

typedef float f32 		__attribute__ ((aligned(SIMD_ALIGNMENT)));
typedef float f32avx	__attribute__ ((aligned(AVX_ALIGNMENT)));
typedef __m128 	float128;
typedef __m256	float256;
typedef __m512	float512;

f32 sse_hvadd(float128 v);
f32 sse3_hvadd(float128 v);
f32 avx_hvadd(float256 v);
f32avx avx512_hvadd(float512 v);

void hvadd_main(int argc, char *argv[]);

// Debug
void print128(float128 v);
void print256(float256 v);

#endif