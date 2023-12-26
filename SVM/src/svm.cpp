#include"svm.h"

double mult(my_vector& x, my_vector& y) {
	int size = x.size();
	assert(y.size() == size);
	double out = 0;
	for (int i = 0; i < size; ++i) {
		out += x[i] * y[i];
	}
	return out;
}

