#include"kmeans.h"

/**
* @brief calculate distance between x and y
*
* @return (double) distance between 2 point x and y.calculate distanc
**/

double KMeans::squareDistance(my_vector& x, my_vector& y) {
	int size = x.size();
	assert(size == y.size());
	double output = 0.0, tmp = 0.0;
	for (int i = 0; i < size; ++i) {
		tmp = x[i] - y[i];
		output += tmp * tmp;
	}
	//output = sqrt(output);
	return output;
}
