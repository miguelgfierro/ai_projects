/*
*
* Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the Delft University of Technology.
* 4. Neither the name of the Delft University of Technology nor the names of
*    its contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
* EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
* BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
*
*/



#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdint.h>
#include "vptree.h"
#include "sptree.h"
#include "t_sne_gpu.h"
#include "dev_array.h"
#include "EuclideanDistances.h"
#include "kernels.h"
#include "check_cuda_device.cuh"

using namespace std;

// Perform t-SNE
void TSNE::run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, double eta, int iterations, float gpu_mem, int verbose, int* landmarks) {
	setbuf(stdout, NULL);

	// Determine whether we are using an exact algorithm
	if (N - 1 < 3 * perplexity) {
		printf("Perplexity ( = %i) too large for the number of data points (%i)!\n", perplexity, N);
		exit(1);
	}
	if (verbose > 0) printf("Using no_dims = %d, perplexity = %f, learning rate = %f, and theta = %f\n", no_dims, perplexity, eta, theta);

	bool exact = (theta == .0) ? true : false;

	// Set learning parameters
	float total_time = .0;
	clock_t start, end;
	int max_iter = iterations, stop_lying_iter = 250, mom_switch_iter = 250;
	double momentum = .5;
	double final_momentum = .8;
	float exageration = 20.0;

	// Allocate some memory
	double* dY = (double*)malloc(N * no_dims * sizeof(double));
	double* uY = (double*)malloc(N * no_dims * sizeof(double));
	double* gains = (double*)malloc(N * no_dims * sizeof(double));
	if (dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int i = 0; i < N * no_dims; i++)    uY[i] = .0;
	for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

	EuclideanDistances* all_euclidean_distances = new EuclideanDistances();

	// Normalize input data (to prevent numerical problems)
	if (verbose > 1) printf("Computing input similarities...\n");

	start = clock();
	// Compute input similarities for exact t-SNE
	double* P = NULL;
	unsigned int* row_P = NULL;
	unsigned int* col_P = NULL;
	double* val_P = NULL;
	if (exact) {

		// Compute similarities
		if (verbose > 0) printf("Exact\n");
		P = (double*)malloc(N * N * sizeof(double));
		if (P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		computeGaussianPerplexity(X, N, D, P, perplexity);

		// Symmetrize input similarities
		if (verbose > 1) printf("Symmetrizing...\n");
		int nN = 0;
		for (int n = 0; n < N; n++) {
			int mN = 0;
			for (int m = n + 1; m < N; m++) {
				P[nN + m] += P[mN + n];
				P[mN + n] = P[nN + m];
				mN += N;
			}
			nN += N;
		}
		double sum_P = .0;
		for (int i = 0; i < N * N; i++) sum_P += P[i];
		for (int i = 0; i < N * N; i++) P[i] /= sum_P;
	}

	// Compute input similarities for approximate t-SNE
	else {

		// Compute asymmetric pairwise input similarities
		computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int)(3 * perplexity), gpu_mem, verbose);

		// Symmetrize input similarities
		symmetrizeMatrix(&row_P, &col_P, &val_P, N);
		double sum_P = .0;
		for (int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
		for (int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
	}
	end = clock();

	// Lie about the P-values
	if (exact) { for (int i = 0; i < N * N; i++)    P[i] *= exageration; }
	else { for (int i = 0; i < row_P[N]; i++)		val_P[i] *= 12.0; }

	// Initialize solution (randomly)
	for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;

	// Perform main training loop
	if (exact && verbose > 1) printf("Input similarities computed in %4.2f seconds\n", (float)(end - start) / CLOCKS_PER_SEC);
	if (!exact && verbose > 1) printf("Input similarities computed in %4.2f seconds (sparsity = %f)\n", (float)(end - start) / CLOCKS_PER_SEC, (double)row_P[N] / ((double)N * (double)N));
	if (verbose > 0) printf("\nLearning embedding...\n");
	start = clock();
	for (int iter = 0; iter < max_iter; iter++) {

		// Compute (approximate) gradient
		if (exact) computeExactGradient(P, Y, N, no_dims, dY);
		else computeGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta); // Why need P?

		// Update gains
		//for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
		for (int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .05) : (gains[i] * .95);
		for (int i = 0; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;

		// Perform gradient update (with momentum and gains)
		for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for (int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

		// Make solution zero-mean
		zeroMean(Y, N, no_dims);

		// Stop lying about the P-values after a while, and switch momentum
		if (iter == stop_lying_iter) {
			if (exact) { for (int i = 0; i < N * N; i++)        P[i] /= exageration; }
			else      { for (int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0; }
		}
		if (iter == mom_switch_iter) momentum = final_momentum;

		// Save tSNE progress after each iteration
		if (verbose > 2)
		{
			// Open file, write first 2 integers and then the data
			FILE *h;
			const int MAX_PATH = 256;
			char interim_filename[MAX_PATH];
			sprintf(interim_filename, "interim_%06i.dat", iter);
			if ((h = fopen(interim_filename, "w + b")) == NULL) {
				printf("Error: could not open data file.\n");
				return;
			}
			fwrite(&N, sizeof(int), 1, h);
			fwrite(&no_dims, sizeof(int), 1, h);
			fwrite(Y, sizeof(double), N * no_dims, h);
			fwrite(landmarks, sizeof(int), N, h);
			fclose(h);
			//printf("Wrote the %ith interim data matrix successfully!\n", iter);
		}

		// Print out progress
		if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
			end = clock();
			double C = .0;
			if (exact) C = evaluateError(P, Y, N, no_dims);
			else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
			if (iter == 0){
				if (verbose > 1) printf("Iteration %d: error is %f\n", iter + 1, C);
			}
			else {
				total_time += (float)(end - start) / CLOCKS_PER_SEC;
				if (verbose > 1) printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float)(end - start) / CLOCKS_PER_SEC);
			}
			start = clock();
		}
	}
	end = clock(); total_time += (float)(end - start) / CLOCKS_PER_SEC;

	// Clean up memory
	free(dY);
	free(uY);
	free(gains);
	if (exact) free(P);
	else {
		free(row_P); row_P = NULL;
		free(col_P); col_P = NULL;
		free(val_P); val_P = NULL;
	}
	if (verbose > 0) printf("Fitting performed in %4.2f seconds.\n", total_time);
}




void TSNE::runWithCenterOfMass(double* data, double* tsne_results, int_least64_t all_data_samples,
	int_least64_t tsned_data_samples, int_least64_t data_space_dims, int_least64_t tsne_space_dims,
	double perplexity, double theta, double eta, int_least64_t iterations, float gpu_mem, int verbose, int* landmarks){

	//Cut the data to be t-sned
	double* data_to_tsne = (double*)malloc(tsned_data_samples * data_space_dims * sizeof(double));
	for (int i = 0; i < (tsned_data_samples* data_space_dims); ++i){ data_to_tsne[i] = data[i]; }

	double* actual_tsne_results = (double*)malloc(tsned_data_samples * tsne_space_dims * sizeof(double));
	run(data_to_tsne, tsned_data_samples, data_space_dims, actual_tsne_results, tsne_space_dims, perplexity, theta, eta, iterations, gpu_mem, verbose, landmarks);
	//run(data, all_data_samples, data_space_dims, tsne_results, tsne_space_dims, perplexity, theta, eta, iterations, gpu_mem);

	for (int i = 0; i < tsned_data_samples * tsne_space_dims; ++i){ tsne_results[i] = actual_tsne_results[i]; }

	int remaining_samples = all_data_samples - tsned_data_samples;

	if (remaining_samples > 0 && verbose > 0)
		printf("\nStarting placemenmt of %i remaining samples on the t-sne space through center of mass computation\n", remaining_samples);

	int iterations = ceil((float)remaining_samples / (float)tsned_data_samples);
	int this_iter = 1;
	while (remaining_samples > 0){

		int samples_in_this_iteration = tsned_data_samples;
		int current_starting_sample = all_data_samples - remaining_samples;

		double* extra_data = (double*)malloc(samples_in_this_iteration * data_space_dims * sizeof(double));

		//Do this to make sure the extra_data has the same size with the data_to_tsne (by putting 0 if we have run out of data)
		//so that we can calculate their euclidean differences on the GPU.
		//The distances between the zeros and the data_to_tsne samples will then be ignored
		if (remaining_samples < samples_in_this_iteration){
			for (int i = 0; i < remaining_samples * data_space_dims; ++i){ extra_data[i] = data[i + current_starting_sample * data_space_dims]; }
			for (int i = remaining_samples * data_space_dims; i < samples_in_this_iteration * data_space_dims; ++i){
				extra_data[i] = 0;
			}
		}
		else{
			for (int i = 0; i < samples_in_this_iteration * data_space_dims; ++i){ extra_data[i] = data[i + current_starting_sample * data_space_dims]; }
		}

		EuclideanDistances* distances = new EuclideanDistances();
		computeSquaredEuclideanDistanceOnGpu(data_to_tsne, extra_data, distances, samples_in_this_iteration, data_space_dims, gpu_mem, 0);
		vector<vector<float> >* all_dist = distances->get_all_euclidean_distances();

		int number_of_closest_samples_to_use = 5;
		if (samples_in_this_iteration > remaining_samples)
			samples_in_this_iteration = remaining_samples;
		for (int sample = 0; sample < samples_in_this_iteration; ++sample){
			vector<float> sample_dists = all_dist->at(sample);
			vector<int> sorted_dist_indices(samples_in_this_iteration);
			vector<float> closest_sorted_dist(number_of_closest_samples_to_use);
			size_t n(0);
			generate(std::begin(sorted_dist_indices), std::end(sorted_dist_indices),
				[&]{ return n++; });

			sort(std::begin(sorted_dist_indices), std::end(sorted_dist_indices),
				[&](int i1, int i2) { return sample_dists[i1] < sample_dists[i2]; });

			for (int i = 0; i < number_of_closest_samples_to_use; ++i){
				closest_sorted_dist[i] = sample_dists[sorted_dist_indices[i]];
			}
			vector<float> weights_of_closest_samples(number_of_closest_samples_to_use);
			for (int i = 0; i < number_of_closest_samples_to_use; ++i){
				float minimum = min_element(closest_sorted_dist.begin(), closest_sorted_dist.end())[0];
				float maximum = max_element(closest_sorted_dist.begin(), closest_sorted_dist.end())[0];
				float X_std = (closest_sorted_dist[i] - minimum) / (maximum - minimum);
				weights_of_closest_samples[i] = X_std * (-1) + 1;
			}

			for (int d = 0; d < tsne_space_dims; ++d){
				float temp_tsne_result = 0;
				for (int i = 0; i < number_of_closest_samples_to_use; ++i){
					temp_tsne_result += actual_tsne_results[sorted_dist_indices[i] * tsne_space_dims + d];// *weights_of_closest_samples[i];
				}
				tsne_results[(current_starting_sample + sample)*tsne_space_dims + d] = temp_tsne_result / number_of_closest_samples_to_use;
			}

			weights_of_closest_samples.clear();
			closest_sorted_dist.clear();
			sorted_dist_indices.clear();
			sample_dists.clear();
		}

		remaining_samples -= tsned_data_samples;

		all_dist->clear();
		free(distances);
		if (verbose > 1) printf("Finished %i iteration of %i of placing extra points on t-sne space\n", this_iter, iterations);
		this_iter += 1;
	}
	free(actual_tsne_results); actual_tsne_results = NULL;

}






// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{
	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;
	double* pos_f = (double*)calloc(N * D, sizeof(double));
	double* neg_f = (double*)calloc(N * D, sizeof(double));
	if (pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }


	tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
	for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

	// Compute final t-SNE gradient
	for (int i = 0; i < N * D; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);
	delete tree;
}

// Compute gradient of the t-SNE cost function (exact)
void TSNE::computeExactGradient(double* P, double* Y, int N, int D, double* dC) {

	// Make sure the current gradient contains zeros
	for (int i = 0; i < N * D; i++) dC[i] = 0.0;

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(Y, N, D, DD);

	// Compute Q-matrix and normalization sum
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	// Perform the computation of the gradient
	nN = 0;
	int nD = 0;
	for (int n = 0; n < N; n++) {
		int mD = 0;
		for (int m = 0; m < N; m++) {
			if (n != m) {
				double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
				for (int d = 0; d < D; d++) {
					dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
				}
			}
			mD += D;
		}
		nN += N;
		nD += D;
	}

	// Free memory
	free(DD); DD = NULL;
	free(Q);  Q = NULL;
}


// Evaluate t-SNE cost function (exactly)
double TSNE::evaluateError(double* P, double* Y, int N, int D) {

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	double* Q = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(Y, N, D, DD);

	// Compute Q-matrix and normalization sum
	int nN = 0;
	double sum_Q = DBL_MIN;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
			else Q[nN + m] = DBL_MIN;
		}
		nN += N;
	}
	for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	// Sum t-SNE error
	double C = .0;
	for (int n = 0; n < N * N; n++) {
		C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	}

	// Clean up memory
	free(DD);
	free(Q);
	return C;
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)
{

	// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double* buff = (double*)calloc(D, sizeof(double));
	double sum_Q = .0;
	for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
	}

	// Clean up memory
	free(buff);
	delete tree;
	return C;
}


// Compute input similarities with a fixed perplexity
void TSNE::computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity) {

	// Compute the squared Euclidean distance matrix
	double* DD = (double*)malloc(N * N * sizeof(double));
	if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
	int nN = 0;
	for (int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX;
		double tol = 1e-5;
		double sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		while (!found && iter < 200) {

			// Compute Gaussian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			double H = 0.0;
			for (int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if (Hdiff > 0) {
					min_beta = beta;
					if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row normalize P
		for (int m = 0; m < N; m++) P[nN + m] /= sum_P;
		nN += N;
	}

	// Clean up memory
	free(DD); DD = NULL;
}


// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P,
	double perplexity, int K, float gpu_mem, int verbose) {
	float start;
	float end;

	if (perplexity > K) printf("Perplexity should be lower than K!\n");

	// Allocate the memory we need
	*_row_P = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
	*_col_P = (unsigned int*)calloc(N * K, sizeof(unsigned int));
	*_val_P = (double*)calloc(N * K, sizeof(double));
	if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	unsigned int* row_P = *_row_P;
	unsigned int* col_P = *_col_P;
	double* val_P = *_val_P;
	double* cur_P = (double*)malloc((N - 1) * sizeof(double));
	if (cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	row_P[0] = 0;
	for (int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int)K;

	// Build ball tree on data set
	VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
	vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
	for (int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
	tree->create(obj_X);

	//If using gpu then calculate and save all euclidean distances and use those to search the tree
	EuclideanDistances* all_distances = new EuclideanDistances();
	if (gpu_mem > 0){
		if (gpu_mem > 1){
			printf("Requested GPU memory needs to be between 0 and 1 (of total available memory). Setting it to 0.8.");
			gpu_mem = 0.8;
		}
		start = clock();
		computeSquaredEuclideanDistanceOnGpu(X, X, all_distances, N, D, gpu_mem, verbose);
		tree->set_all_euclidean_distances(*all_distances);
		end = clock();
		if (verbose > 1) printf("Time spent in calculating all distances in GPU: %f\n", float(end - start) / CLOCKS_PER_SEC);
	}
	else{
		if (verbose > 0) printf("Using CPU to calculate distances during tree search\n");
	}

	// Loop over all points to find nearest neighbors
	if (verbose > 0) printf("Building tree...\n");
	vector<DataPoint> indices;
	vector<double> distances;

	start = clock();
	for (int n = 0; n < N; n++) {
		if (n % 10000 == 0 && verbose > 1) printf(" - Building tree and finding perplexities, point %d of %d\n", n, N);
		// Find nearest neighbors
		indices.clear();
		distances.clear();
		tree->search(obj_X[n], K + 1, &indices, &distances);

		// Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while (!found && iter < 200) {

			// Compute Gaussian kernel row
			for (int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1]);

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < K; m++) sum_P += cur_P[m];
			double H = .0;
			for (int m = 0; m < K; m++) H += beta * (distances[m + 1] * cur_P[m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if (Hdiff > 0) {
					min_beta = beta;
					if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize current row of P and store in matrix
		for (unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
		for (unsigned int m = 0; m < K; m++) {
			col_P[row_P[n] + m] = (unsigned int)indices[m + 1].index();
			val_P[row_P[n] + m] = cur_P[m];
		}
		distances.clear();
		indices.clear();
	}
	end = clock();
	if (verbose > 1) printf("Time spent building tree and finding perplexities: %f\n", float(end - start) / CLOCKS_PER_SEC);

	// Clean up memory
	obj_X.clear();
	free(cur_P);
	delete tree;
	delete all_distances;
}


// Symmetrizes a sparse matrix
void TSNE::symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

	// Get sparse matrix
	unsigned int* row_P = *_row_P;
	unsigned int* col_P = *_col_P;
	double* val_P = *_val_P;

	// Count number of elements and row counts of symmetric matrix
	int* row_counts = (int*)calloc(N, sizeof(int));
	if (row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int i = row_P[n]; i < row_P[n + 1]; i++) {

			// Check whether element (col_P[i], n) is present
			bool present = false;
			for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if (col_P[m] == n) present = true;
			}
			if (present) row_counts[n]++;
			else {
				row_counts[n]++;
				row_counts[col_P[i]]++;
			}
		}
	}
	int no_elem = 0;
	for (int n = 0; n < N; n++) no_elem += row_counts[n];

	// Allocate memory for symmetrized matrix
	unsigned int* sym_row_P = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
	unsigned int* sym_col_P = (unsigned int*)malloc(no_elem * sizeof(unsigned int));
	double* sym_val_P = (double*)malloc(no_elem * sizeof(double));
	if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	// Construct new row indices for symmetric matrix
	sym_row_P[0] = 0;
	for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int)row_counts[n];

	// Fill the result matrix
	int* offset = (int*)calloc(N, sizeof(int));
	if (offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

			// Check whether element (col_P[i], n) is present
			bool present = false;
			for (unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if (col_P[m] == n) {
					present = true;
					if (n <= col_P[i]) {                                                 // make sure we do not add elements twice
						sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
						sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
						sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
						sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
					}
				}
			}


			// If (col_P[i], n) is not present, there is no addition involved
			if (!present) {
				sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
				sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
				sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
				sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
			}


			// Update offsets
			if (!present || (present && n <= col_P[i])) {
				offset[n]++;
				if (col_P[i] != n) offset[col_P[i]]++;
			}
		}
	}

	// Divide the result by two
	for (int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

	// Return symmetrized matrices
	free(*_row_P); *_row_P = sym_row_P;
	free(*_col_P); *_col_P = sym_col_P;
	free(*_val_P); *_val_P = sym_val_P;

	// Free up some memery
	free(offset); offset = NULL;
	free(row_counts); row_counts = NULL;
}

// Compute squared Euclidean distance matrix (using BLAS)
void TSNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
	double* dataSums = (double*)calloc(N, sizeof(double));
	if (dataSums == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	int nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			dataSums[n] += (X[nD + d] * X[nD + d]);
		}
		nD += D;
	}
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			DD[nN + m] = dataSums[n] + dataSums[m];
		}
		nN += N;
	}
	nN = 0; nD = 0;
	for (int n = 0; n < N; n++) {
		int mD = 0;
		DD[nN + n] = 0.0;
		for (int m = n + 1; m < N; m++) {
			DD[nN + m] = 0.0;
			for (int d = 0; d < D; d++) {
				DD[nN + m] += (X[nD + d] - X[mD + d]) * (X[nD + d] - X[mD + d]);
			}
			DD[m * N + n] = DD[nN + m];
			mD += D;
		}
		nN += N; nD += D;
	}
	free(dataSums); dataSums = NULL;
}

//Compute all squared euclidean distances between all samples in data_X and data_Y on the GPU and store them in a EuclideanDistances object
//Both data matrices must have the same size N_samples X D_features 
//If the GPU cannot hold in memory N*N distances then calculate those incrementaly:
//For every iteration calculate on the GPU all N*N distances but put on the GPU memory only a portion.
//Then copy that portion onto a CPU 2D vector in the correct place (this is done in the EuclideanDistances.set_all_euclidean_distances function).
//It is assumed that there is enough RAM to hold N*N floats
void TSNE::computeSquaredEuclideanDistanceOnGpu(double* data_X, double* data_Y, EuclideanDistances* distances, int_least64_t N, int_least64_t D,
	float gpu_mem, int verbose){
	int_least64_t size_in = N * D;

	int num_of_cuda_devices = 0;
	initialize_cuda_device(&num_of_cuda_devices, verbose);

	double free_db, used_db, total_db, cuda_memory_needed, memory_to_use;
	get_device_memory(&free_db, &used_db, &total_db);

	//Figure out available GPU memory and plan the number of iterations 
	//so that every iteration uses only an available amount of memory
	memory_to_use = gpu_mem * free_db;
	cuda_memory_needed = (double(4) * (double)N * (double)N);
	int num_of_iters = ceil(cuda_memory_needed / memory_to_use);

	int_least64_t total_distances = N * N;
	int_least64_t* distances_per_iter = (int_least64_t*)malloc(num_of_iters * sizeof(int_least64_t));
	distances_per_iter[0] = total_distances;
	if (num_of_iters > 1){
		for (int iter = 0; iter < num_of_iters; ++iter){
			distances_per_iter[iter] = memory_to_use / sizeof(float);
			if (iter == num_of_iters - 1){
				distances_per_iter[iter] = N * N - iter * distances_per_iter[iter - 1];
			}
		}
	}

	// Allocate input matrix memory on the host
	float* h_in_X;
	h_in_X = (float*)malloc(size_in * sizeof(float));
	float* h_in_Y;
	h_in_Y = (float*)malloc(size_in * sizeof(float));

	//Initialise matrices on the host (casting data from double to float to allow the GPU to do work)
	for (int i = 0; i<N; i++){
		for (int j = 0; j<D; j++){
			h_in_X[i*D + j] = (float)data_X[i*D + j];
			h_in_Y[i*D + j] = (float)data_Y[i*D + j];
		}
	}

	int_least64_t distances_left = total_distances;
	for (int iter = 0; iter < num_of_iters; ++iter){

		int_least64_t start_out = total_distances - distances_left;
		int_least64_t end_out = start_out + distances_per_iter[iter];

		int_least64_t size_out = distances_per_iter[iter];
		if (verbose > 1) printf("GPU iteration = %i, distance elements calculated = %lld\n", iter, size_out);

		// Allocate partial distances matrix memory on the host
		float* h_out;
		h_out = (float*)malloc(size_out * sizeof(float));

		//Allocate memory on the device 
		dev_array<float> d_out(size_out);
		dev_array<float> d_in_X(size_in);
		dev_array<float> d_in_Y(size_in);
		if (verbose > 1) print_device_memory();

		//Copy to device
		d_in_X.set(&h_in_X[0], size_in);
		d_in_Y.set(&h_in_Y[0], size_in);

		//Do the GPU calculations
		//matrixEuclideanDistanceFast(d_in_X.getData(), d_out.getData(), N, D, start_out, end_out);
		matrixEuclideanDistanceFast(d_in_X.getData(), d_in_Y.getData(), d_out.getData(), N, D, start_out, end_out);
		cudaDeviceSynchronize();

		//Get data from the device
		d_out.get(&h_out[0], size_out);
		cudaDeviceSynchronize();

		//Put computed distances into the EuclideanDistances object
		distances->set_all_euclidean_distances(h_out, (int_least64_t)N, start_out, end_out);

		free(h_out);

		distances_left -= distances_per_iter[iter];
	}
	free(h_in_X);
	free(h_in_Y);
	free(distances_per_iter);
}


// Makes data zero-mean
void TSNE::zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*)calloc(D, sizeof(double));
	if (mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	int nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
		nD += D;
	}
	for (int d = 0; d < D; d++) {
		mean[d] /= (double)N;
	}

	// Subtract data mean
	nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
		nD += D;
	}
	free(mean); mean = NULL;
}

void TSNE::normalize(double* X, int N, int D){
	double max_X = .0;
	for (int i = 0; i < N * D; i++) {
		if (X[i] > max_X) max_X = X[i];
	}
	for (int i = 0; i < N * D; i++) X[i] /= max_X;
}


// Generates a Gaussian random number
double TSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while ((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity,
	double* eta, int* iterations, int* seed, float* gpu_mem, int* verbose, int* rand_seed) {

	// Open file, read first 2 integers, allocate memory, and read the data
	FILE *h;
	if ((h = fopen("data.dat", "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
	fread(n, sizeof(int), 1, h);											// number of datapoints
	fread(d, sizeof(int), 1, h);											// original dimensionality
	fread(theta, sizeof(double), 1, h);										// gradient accuracy
	fread(perplexity, sizeof(double), 1, h);								// perplexity
	fread(eta, sizeof(double), 1, h);										// eta (learning rate)
	fread(no_dims, sizeof(int), 1, h);										// output dimensionality
	fread(iterations, sizeof(int), 1, h);									// number of iterations
	fread(seed, sizeof(int), 1, h);											// number of samples to t-sne
	fread(gpu_mem, sizeof(float), 1, h);									// percentage of gpu memory to use (if 0 no gpu is used)
	fread(verbose, sizeof(int), 1, h);										// verbosity (between 0 and 2)
	*data = (double*)malloc(*d * *n * sizeof(double));
	if (*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	fread(*data, sizeof(double), *n * *d, h);                               // the data
	if (!feof(h)) fread(rand_seed, sizeof(int), 1, h);                      // random seed
	fclose(h);
	if (*verbose > 0) printf("Read the %i x %i data matrix successfully!\n", *n, *d);
	return true;
}

// Function that saves map to a t-SNE file
void TSNE::save_data(double* data, int* landmarks, double* costs, int n, int d, int verbose) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	if ((h = fopen("result.dat", "w+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
	fwrite(data, sizeof(double), n * d, h);
	fwrite(landmarks, sizeof(int), n, h);
	fwrite(costs, sizeof(double), n, h);
	fclose(h);
	if (verbose > 0) printf("Wrote the %i x %i data matrix successfully!\n\n", n, d);
}


// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

	// Define some variables
	int origN, N, D, no_dims, iterations, *landmarks;
	double perc_landmarks;
	double perplexity, theta, eta, *data;
	float gpu_mem;
	int verbose;
	int rand_seed = -1;
	int seed = 0;
	TSNE* tsne = new TSNE();

	time_t start = clock();
	// Read the parameters and the dataset
	if (tsne->load_data(&data, &origN, &D, &no_dims, &theta, &perplexity, &eta, &iterations, &seed, &gpu_mem, &verbose, &rand_seed)) {

		// Set random seed
		if (rand_seed >= 0) {
			if (verbose > 0) printf("Using random seed: %d\n", rand_seed);
			srand((unsigned int)rand_seed);
		}
		else {
			if (verbose > 0) printf("Using current time as random seed...\n");
			srand(time(NULL));
		}

		// Make dummy landmarks
		N = origN;
		int* landmarks = (int*)malloc(N * sizeof(int));
		if (landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		for (int n = 0; n < N; n++) landmarks[n] = n;

		// Now fire up the SNE implementation
		double* Y = (double*)malloc(N * no_dims * sizeof(double));
		double* costs = (double*)calloc(N, sizeof(double));
		if (Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }

		tsne->zeroMean(data, N, D);
		tsne->normalize(data, N, D);
		if (seed == 0){
			seed = origN;
		}
		tsne->runWithCenterOfMass(data, Y, origN, seed, D, no_dims, perplexity, theta, eta, iterations, gpu_mem, verbose, landmarks);

		// Save the results
		tsne->save_data(Y, landmarks, costs, N, no_dims, verbose);

		// Clean up the memory
		free(data); data = NULL;
		free(Y); Y = NULL;
		free(costs); costs = NULL;
		free(landmarks); landmarks = NULL;
	}
	delete(tsne);
	time_t end = clock();
	if (verbose > 0) printf("T-sne required %f seconds (%f minutes) to run\n", float(end - start) / CLOCKS_PER_SEC, float(end - start) / (60 * CLOCKS_PER_SEC));

}