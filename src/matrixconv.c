#include "matrixconv.h"

void scalar_matrix_conv3x3(f32 **m, f32 **k, f32 **dest, int N){

	int i,j,k,l;

	for(i=1; i<N-1; i++){

		for(j=1; j<N-1; j++){

			dest[i][j] = 	m[i-1][j-1] * k[0][0] + m[i-1][j] * k[0][1] + m[i-1][j+1] * k[0][2] +
							m[i][j-1]   * k[1][0] + m[i][j]	  * k[1][1] + m[i][j+1]   * k[1][2] +
							m[i+1][j-1] * k[2][0] + m[i+1][k] * k[2][1] + m[i+1][j+1] * k[2][2];

		}

	}

}

void matrixconv_main(int argc, char *argv[]){

	f32 **m;
	f32 **k;
	f32 **dest;


	
}