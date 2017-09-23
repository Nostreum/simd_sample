#include "hvadd.h"

#ifdef __SSE__

f32 sse_hvadd(float128 v){

	float128 v_shuffled = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1));
	float128 v_adds	= _mm_add_ps(v, v_shuffled);
	v_shuffled = _mm_shuffle_ps(v_adds, v_adds, _MM_SHUFFLE(1,0,3,2));
	v_adds = _mm_add_ps(v_adds, v_shuffled);

	return _mm_cvtss_f32(v_adds);
}

#endif

#ifdef __SSE2__

f32 sse3_hvadd(float128 v){

	float128 v_shuffled = _mm_movehdup_ps(v);
	float128 v_adds	= _mm_add_ps(v, v_shuffled);
	v_shuffled = _mm_shuffle_ps(v_adds, v_adds, _MM_SHUFFLE(1,0,3,2));
	v_adds = _mm_add_ps(v_adds, v_shuffled);

	return _mm_cvtss_f32(v_adds);
}

#endif

#ifdef __AVX__

f32avx avx_hvadd(float256 v){

	float128 v_low = _mm256_castps256_ps128(v);
	float128 v_high = _mm256_extractf128_ps(v, 1); 
	v_low = _mm_add_ps(v_low, v_high);
	return sse_hvadd(v_low);

}

#endif

void hvadd_main(int argc, char *argv[]){
	
	float128 v = _mm_set_ps(2, 3, 9, 24);
	float256 v2 = _mm256_set_ps(2, 3, 9, 24, 1, 0, 11, 7);

	f32 res_sse = sse_hvadd(v);
	f32 res_sse3 = sse3_hvadd(v);
	f32 res_avx = avx_hvadd(v2);

	print128(v);printf(" => SSE : %f\n", res_sse);
	print128(v);printf(" => SSE3 t: %f\n", res_sse3);
	print256(v2);printf(" => AVX : %f\n", res_avx);
}