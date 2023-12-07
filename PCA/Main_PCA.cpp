#include"PCA.h"
#include <Eigen/Dense>
#include <vector>


#include <iostream>

int main233() {
    
    std::vector<std::vector<double>> testData = {
        {2.5, 2.4, 3.2},
        {0.5, 0.7, 1.2},
        {2.2, 2.9, 3.1},
        {1.9, 2.2, 2.9},
        {3.1, 3.0, 3.3},
    };
	/*
    std::vector<std::vector<double>> testData = {
        {1,2},
        {-1,-2}
    };*/
    int k = 2;
	PCA p(k);
    std::vector<std::vector<double>> reducedData = p.transform(testData, k);

    std::cout << "Reduced Data:" << std::endl;
    for (const auto& row : reducedData) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
	std::vector<std::vector<double>> c = {
		{2.6,2.5,3.3},
		{0.7,0.8,1.3}
	};
	c=p.transform(c, 2);
	for (auto& as : c) {
		for (double i : as) {
			printf("%f ", i);
		}
		printf("\n");
	}
    return 0;
}