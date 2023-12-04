#include"kmeans.h"

int main2333() {
	KMeans km(4, 2);
	std::vector<std::vector<double>> trainPoints = {
		{0.1,0.1},
		{0.2,0.1},
		{0.1,0.2},
		{0.2,0.2},

		{10.1,10.1},
		{10.2,10.1},
		{10.1,10.2},
		{10.2,10.2},

		{5.1,5.1},
		{5.2,5.1},
		{5.2,5.1},
		{5.2,5.2},

		{20.1,20.1},
		{20.2,20.1},
		{20.1,20.2},
		{20.2,20.2},
	};

	km.train(trainPoints, 1000);
	km.printCenters();
	return 0;
}