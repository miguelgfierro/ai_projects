
#include <vector>
#include <stdint.h>

#ifndef EUCLIDEAN_DISTANCES_H
#define EUCLIDEAN_DISTANCES_H


using namespace std;

class EuclideanDistances{

private:
	int _N;
	vector<vector<float> > all_euclidean_distances;

public:
	EuclideanDistances(){
		_N = 0;
		all_euclidean_distances.clear();
	}

	EuclideanDistances(float* eucl_dist_array, int_least64_t n){
		set_all_euclidean_distances(eucl_dist_array, n, 0, n*n);
	}

	EuclideanDistances(float* eucl_dist_array, int_least64_t n, int_least64_t start, int_least64_t end){
		set_all_euclidean_distances(eucl_dist_array, n, start, end);
	}

	~EuclideanDistances(){
		_N = 0;
		all_euclidean_distances.clear();
	}


	void set_all_euclidean_distances(float* ed, int_least64_t n, int_least64_t start, int_least64_t end){

		if (all_euclidean_distances.size() == 0)
			all_euclidean_distances.resize(n);

		for (int_least64_t i = start; i < end; ++i){
			if (all_euclidean_distances[i / n].size() == 0)
				all_euclidean_distances[i / n].resize(n);
			all_euclidean_distances[i / n][i % n] = sqrt(ed[i - start]);
		}
	}

	vector<vector<float> >* get_all_euclidean_distances(){
		return &all_euclidean_distances;
	}
};

#endif // !EUCLIDEAN_DISTANCES_H
