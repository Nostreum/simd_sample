#include "matrixsum.h"

f32 scalar_matrix_sum(f32 **m, int N){

	int i,j;
	float sum = 0.0f;

	for(i=0; i<N; i++)
		for(j=0; j<N; j++)
			sum += m[i][j]; 

	return sum;
}


f32 sse_matrix_sum(f32 **m, int N){

	int i,j;
	f32 sum = 0.0f;

	for(i=0; i<N; i++){
		for(j=0; j<N; j+=4){

			float128 v = _mm_load_ps(&m[i][j]);

			sum += sse_hvadd(v);

		}
	}

	return sum;

}

f32 sse_unrolling_matrix_sum(f32 **m, int N){

	int i,j;
	int border = N%4;
	f32 sum = 0.0f;

	for(i=0; i<N-border; i+=4){
		for(j=0; j<N; j+=4){

			float128 v1 = _mm_load_ps(&m[i][j]);
			float128 v2 = _mm_load_ps(&m[i+1][j]);
			float128 v3 = _mm_load_ps(&m[i+2][j]);
			float128 v4 = _mm_load_ps(&m[i+3][j]);

			sum += sse_hvadd(v1);
			sum += sse_hvadd(v2);
			sum += sse_hvadd(v3);
			sum += sse_hvadd(v4);
		}
	}

	for(i=N-border; i<N; i++){
		for(j=0; j<N; j+=4){

			float128 v1 = _mm_load_ps(&m[i][j]);

			sum += sse_hvadd(v1);
		}
	}

	return sum;

}

f32 sse_unrolling_openmp_matrix_sum(f32 **m, int N){

	int i,j;
	int border = N%4;
	f32 sum = 0.0f;

	#pragma omp for
	for(i=0; i<N-border; i+=4){
		for(j=0; j<N; j+=4){

			float128 v1 = _mm_load_ps(&m[i][j]);
			float128 v2 = _mm_load_ps(&m[i+1][j]);
			float128 v3 = _mm_load_ps(&m[i+2][j]);
			float128 v4 = _mm_load_ps(&m[i+3][j]);

			sum += sse_hvadd(v1);
			sum += sse_hvadd(v2);
			sum += sse_hvadd(v3);
			sum += sse_hvadd(v4);
		}
	}

	#pragma omp for
	for(i=N-border; i<N; i++){
		for(j=0; j<N; j+=4){

			float128 v1 = _mm_load_ps(&m[i][j]);

			sum += sse_hvadd(v1);
		}
	}

	return sum;

}

f32 sse3_matrix_sum(f32 **m, int N){

	int i,j;
	f32 sum = 0.0f;

	for(i=0; i<N; i++){
		for(j=0; j<N; j+=4){

			float128 v = _mm_load_ps(&m[i][j]);

			sum += sse3_hvadd(v);

		}
	}

	return sum;

}

f32 sse3_unrolling_matrix_sum(f32 **m, int N){

	int i,j;
	int border = N%4;
	f32 sum = 0.0f;

	for(i=0; i<N-border; i+=4){
		for(j=0; j<N; j+=4){

			float128 v1 = _mm_load_ps(&m[i][j]);
			float128 v2 = _mm_load_ps(&m[i+1][j]);
			float128 v3 = _mm_load_ps(&m[i+2][j]);
			float128 v4 = _mm_load_ps(&m[i+3][j]);

			sum += sse3_hvadd(v1);
			sum += sse3_hvadd(v2);
			sum += sse3_hvadd(v3);
			sum += sse3_hvadd(v4);
		}
	}

	for(i=N-border; i<N; i++){
		for(j=0; j<N; j+=4){

			float128 v1 = _mm_load_ps(&m[i][j]);

			sum += sse3_hvadd(v1);
		}
	}

	return sum;

}

f32 sse3_unrolling_openmp_matrix_sum(f32 **m, int N){

	int i,j;
	int border = N%4;
	f32 sum = 0.0f;

	#pragma omp for
	for(i=0; i<N-border; i+=4){
		for(j=0; j<N; j+=4){

			float128 v1 = _mm_load_ps(&m[i][j]);
			float128 v2 = _mm_load_ps(&m[i+1][j]);
			float128 v3 = _mm_load_ps(&m[i+2][j]);
			float128 v4 = _mm_load_ps(&m[i+3][j]);

			sum += sse3_hvadd(v1);
			sum += sse3_hvadd(v2);
			sum += sse3_hvadd(v3);
			sum += sse3_hvadd(v4);
		}
	}

	#pragma omp for
	for(i=N-border; i<N; i++){
		for(j=0; j<N; j+=4){

			float128 v1 = _mm_load_ps(&m[i][j]);

			sum += sse3_hvadd(v1);
		}
	}

	return sum;

}

f32 avx_matrix_sum(f32avx **m, int N){

	int i,j;
	f32 sum = 0.0f;

	for(i=0; i<N; i++){
		for(j=0; j<N; j+=8){

			float256 v = _mm256_loadu_ps(&m[i][j]);

			sum += avx_hvadd(v);

		}
	}

	return sum;

}

f32 avx_unrolling_matrix_sum(f32avx **m, int N){

	int i,j;
	int border = N%4;
	f32 sum = 0.0f;

	for(i=0; i<N-border; i+=4){
		for(j=0; j<N; j+=8){

			float256 v1 = _mm256_loadu_ps(&m[i][j]);
			float256 v2 = _mm256_loadu_ps(&m[i+1][j]);
			float256 v3 = _mm256_loadu_ps(&m[i+2][j]);
			float256 v4 = _mm256_loadu_ps(&m[i+3][j]);

			sum += avx_hvadd(v1);
			sum += avx_hvadd(v2);
			sum += avx_hvadd(v3);
			sum += avx_hvadd(v4);
		}
	}

	for(i=N-border; i<N; i++){
		for(j=0; j<N; j+=8){

			float256 v1 = _mm256_loadu_ps(&m[i][j]);

			sum += avx_hvadd(v1);
		}
	}

	return sum;

}

f32 avx_unrolling_openmp_matrix_sum(f32avx **m, int N){

	int i,j;
	int border = N%4;
	f32 sum = 0.0f;

	#pragma omp for
	for(i=0; i<N-border; i+=4){
		for(j=0; j<N; j+=8){

			float256 v1 = _mm256_loadu_ps(&m[i][j]);
			float256 v2 = _mm256_loadu_ps(&m[i+1][j]);
			float256 v3 = _mm256_loadu_ps(&m[i+2][j]);
			float256 v4 = _mm256_loadu_ps(&m[i+3][j]);

			sum += avx_hvadd(v1);
			sum += avx_hvadd(v2);
			sum += avx_hvadd(v3);
			sum += avx_hvadd(v4);
		}
	}

	#pragma omp for
	for(i=N-border; i<N; i++){
		for(j=0; j<N; j+=8){

			float256 v1 = _mm256_loadu_ps(&m[i][j]);

			sum += avx_hvadd(v1);
		}
	}

	return sum;

}

void matrixsum_main(int argc, char *argv[]){

	srand(time(NULL));

	int i,j;
	int N = 4;
	f32 **m;
	f32avx **mavx;

	if(argc>1)
		N = atoi(argv[1]);

	m = malloc(N * sizeof(f32*));
	mavx = malloc(N * sizeof(f32avx*));

	for(i=0; i<N; i++){
		m[i] = malloc(N * sizeof(f32));
		mavx[i] = malloc(N * sizeof(f32avx));
	}

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			float r = rand()%10;
			m[i][j] = r;
			mavx[i][j] = r;
		}
	}

	f32 res_scalar = scalar_matrix_sum(m, N);

	f32 res_sse = sse_matrix_sum(m, N);
	f32 res_sse_unrolling = sse_unrolling_matrix_sum(m, N);
	f32 res_sse_unrolling_omp = sse_unrolling_openmp_matrix_sum(m, N);

	f32 res_sse3 = sse3_matrix_sum(m, N);
	f32 res_sse3_unrolling = sse3_unrolling_matrix_sum(m, N);
	f32 res_sse3_unrolling_omp = sse3_unrolling_openmp_matrix_sum(m, N);

	f32 res_avx = avx_matrix_sum(mavx, N);
	f32 res_avx_unrolling = avx_unrolling_matrix_sum(mavx, N);
	f32 res_avx_unrolling_omp = avx_unrolling_openmp_matrix_sum(mavx, N);

	printf("\n\n=========== SCALAR ================ \n");
	printf("Matrix sum scalar : %.3f \n", res_scalar);

	printf("\n============== SSE ================ \n");
	printf("Matrix sum sse : %.3f \n", res_sse);
	printf("Matrix sum sse + unrolling : %.3f \n", res_sse_unrolling);
	printf("Matrix sum sse + unrolling + omp : %.3f \n", res_sse_unrolling_omp);

	printf("\n============== SSE 2 ============== \n");
	printf("Matrix sum sse3 : %.3f \n", res_sse3);
	printf("Matrix sum sse3 + unrolling : %.3f \n", res_sse3_unrolling);
	printf("Matrix sum sse3 + unrolling + omp : %.3f \n", res_sse3_unrolling_omp);

	printf("\n============== AVX ============== \n");
	printf("Matrix sum avx : %.3f \n", res_avx);
	printf("Matrix sum avx + unrolling : %.3f \n", res_avx_unrolling);
	printf("Matrix sum avx + unrolling + omp : %.3f \n", res_avx_unrolling_omp);


}











