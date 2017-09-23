#ifndef _MATRIX_SUM_H
#define _MATRIX_SUM_H

#include <stdio.h>
#include <time.h>
#include <math.h>

#include "hvadd.h"

void matrixsum_main(int argc, char *argv[]);

f32 scalar_matrix_sum(f32 **m, int N);
f32 sse_matrix_sum(f32 **m, int N);


#endif