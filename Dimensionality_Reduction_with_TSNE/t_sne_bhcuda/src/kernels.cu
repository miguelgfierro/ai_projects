#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "math_functions.h"
#include "check_cuda_device.cuh"
#include "kernels.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>




using namespace std;

__device__ int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x)
		+ threadIdx.x;
	return threadId;
}

__global__ void matrixSquareElementWiseKernel(float* in, float* out, int n, int m){
	extern __shared__ float Rs[];

	int index = getGlobalIdx_2D_2D();
	if (index < n*m){

		out[index] = in[index] * in[index];

	}
}

__global__ void matrixEuclideanDistanceKernelFast(float* in, float* out, int n, int m){
	__shared__ float Ys[16][16];
	__shared__ float Xs[16][16];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int yBegin = by * 16 * m;
	int xBegin = bx * 16 * m;

	int yEnd = yBegin + m - 1, y, x, k, o;

	float tmp, s = 0;

	for (y = yBegin, x = xBegin;
		y <= yEnd;
		y += 16, x += 16){
		Ys[ty][tx] = in[y + ty * m + tx];
		Xs[tx][ty] = in[x + ty * m + tx];
		__syncthreads();

		for (k = 0; k<16; k++){
			tmp = Ys[ty][k] - Xs[k][tx];
			s += tmp * tmp;
		}
		__syncthreads();
	}
	o = by * 16 * n + ty * n + bx * 16 + tx;
	out[o] = s;
}

__global__ void matrixEuclideanDistanceKernelFastPartialOut(float* in, float* out, int_least64_t n, int_least64_t m, int_least64_t start_out, int_least64_t end_out){
	__shared__ float Ys[16][16];
	__shared__ float Xs[16][16];

	int_least64_t bx = blockIdx.x, by = blockIdx.y;
	int_least64_t tx = threadIdx.x, ty = threadIdx.y;

	int_least64_t yBegin = by * 16 * m;
	int_least64_t xBegin = bx * 16 * m;

	int_least64_t yEnd = yBegin + m - 1, y, x, k;
	int_least64_t o;

	float tmp, s = 0;

	for (y = yBegin, x = xBegin;
		y <= yEnd;
		y += 16, x += 16){
		Ys[ty][tx] = in[y + ty * m + tx];
		Xs[tx][ty] = in[x + ty * m + tx];
		__syncthreads();

		for (k = 0; k<16; k++){
			tmp = Ys[ty][k] - Xs[k][tx];
			s += tmp * tmp;
		}
		__syncthreads();
	}

	o = by * 16 * n + ty * n + bx * 16 + tx;
	if (o >= start_out && o < end_out){
		out[o - start_out] = s;
	}
}

__global__ void matrixEuclideanDistanceKernelFastPartialOut(float* in_X, float* in_Y, float* out, int_least64_t n, int_least64_t m, int_least64_t start_out, int_least64_t end_out){
	__shared__ float Ys[16][16];
	__shared__ float Xs[16][16];

	int_least64_t bx = blockIdx.x, by = blockIdx.y;
	int_least64_t tx = threadIdx.x, ty = threadIdx.y;

	int_least64_t yBegin = by * 16 * m;
	int_least64_t xBegin = bx * 16 * m;

	int_least64_t yEnd = yBegin + m - 1, y, x, k;
	int_least64_t o;

	float tmp, s = 0;

	for (y = yBegin, x = xBegin;
		y <= yEnd;
		y += 16, x += 16){
		Ys[ty][tx] = in_Y[y + ty * m + tx];
		Xs[tx][ty] = in_X[x + ty * m + tx];
		__syncthreads();

		for (k = 0; k<16; k++){
			tmp = Ys[ty][k] - Xs[k][tx];
			s += tmp * tmp;
		}
		__syncthreads();
	}

	o = by * 16 * n + ty * n + bx * 16 + tx;
	if (o >= start_out && o < end_out){
		out[o - start_out] = s;
	}
}

__global__ void matrixEuclideanDistanceKernel(float* in, float* out, int n, int m){
	extern __shared__ float Rs[];
	float tmp, s;
	int myRow = blockIdx.x*blockDim.x + threadIdx.x;
	for (int r = 0; r<n; r++){ //outer loop
		s = 0;
		for (int i = 0; i <= m / 256; i++){
			if (i * 256 + threadIdx.x < m)
				Rs[i * 256 + threadIdx.x] = in[r*m + i * 256 + threadIdx.x];
		}
		__syncthreads();
		for (int i = 0; i<m && myRow<n; i++){
			tmp = Rs[i] - in[myRow*m + i];
			s += tmp*tmp;
		}
		if (myRow<n)
			out[myRow*n + r] = s;
		__syncthreads();
	}
}


void matrixEuclideanDistanceFast(float *in, float *out, int n, int m){
	dim3 block(16, 16);
	dim3 grid(ceil((float)n / (float)16), ceil((float)n / (float)16));

	matrixEuclideanDistanceKernelFast << <grid, block >> >(in, out, n, m);
	exit_on_cuda_error("matrixEuclideanDistanceKernelFast");
}

void matrixEuclideanDistanceFast(float* in, float* out, int n, int m, int_least64_t start_out, int_least64_t end_out){
	dim3 block(16, 16);
	dim3 grid(ceil((float)n / (float)16), ceil((float)n / (float)16));

	matrixEuclideanDistanceKernelFastPartialOut << <grid, block >> >(in, out, (int_least64_t)n, (int_least64_t)m, start_out, end_out);
	exit_on_cuda_error("matrixEuclideanDistanceKernelFastPartialOut");
}

void matrixEuclideanDistanceFast(float* in_X, float* in_Y, float* out, int n, int m, int_least64_t start_out, int_least64_t end_out){
	dim3 block(16, 16);
	dim3 grid(ceil((float)n / (float)16), ceil((float)n / (float)16));

	matrixEuclideanDistanceKernelFastPartialOut << <grid, block >> >(in_X, in_Y, out, (int_least64_t)n, (int_least64_t)m, start_out, end_out);
	exit_on_cuda_error("matrixEuclideanDistanceKernelFastPartialOut");
}

void matrixEuclideanDistance(float *in, float *out, int n, int m){
	dim3 threadsPerBlock(256, 1, 1);
	dim3 blocksPerGrid(ceil((float)n / (float)256), 1, 1);
	int size_of_shared = m * sizeof(float);
	matrixEuclideanDistanceKernel << <blocksPerGrid, threadsPerBlock, size_of_shared >> >(in, out, n, m);
	exit_on_cuda_error("matrixEuclideanDistance");
}

void matrixSquareElementWise(float* in, float* out, int n, int m){
	dim3 block(16, 16);
	dim3 grid(ceil((float)n / (float)16), ceil((float)n / (float)16));

	matrixSquareElementWiseKernel << <grid, block >> >(in, out, n, m);
	exit_on_cuda_error("matrixSquareElementWise");
}
