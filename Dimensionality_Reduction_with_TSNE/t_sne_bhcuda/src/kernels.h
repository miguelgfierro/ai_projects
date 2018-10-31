

#include <stdint.h>

#ifndef KERNELS_CUH
#define KERNELS_CUH

void matrixEuclideanDistanceFast(float *in, float *out, int n, int m);
void matrixEuclideanDistanceFast(float* in, float* out, int n, int m, int_least64_t start_out, int_least64_t end_out);
void matrixEuclideanDistance(float *in, float *out, int n, int m);
void matrixSquareElementWise(float* in, float* out, int n, int m);
void matrixEuclideanDistanceFast_DebugToFile(float* in, float* out, int n, int m, int_least64_t start_out, int_least64_t end_out);
void matrixEuclideanDistanceFast(float* in_X, float* in_Y, float* out, int n, int m, int_least64_t start_out, int_least64_t end_out);
#endif