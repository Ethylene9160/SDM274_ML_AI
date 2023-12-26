#include"kmeans.h"

int main2333333() {
	KMeans km(4, 2);
	std::vector<std::vector<double>> trainPoints = {
		{0.1,0.1},
		{0.2,0.1},
		{0.1,0.2},
		{0.2,0.2},
		
		{2.2,0.1},
		{2.1,0.2},
		{2.2,0.2},
		{2.1,0.1},

		{0.2,2.1},
		{0.1,2.2},
		{0.2,2.2},
		{0.1,2.1},

		{2.2,2.1},
		{2.1,2.2},
		{2.2,2.2},
		{2.1,2.1},
	};

	km.train(trainPoints, 20000);
	km.printCenters();
	return 0;
}