#include"PCA.h"
#include <Eigen/Dense>
#include <vector>


#include <iostream>

int main() {
    /*
    std::vector<std::vector<double>> testData = {
        {2.5, 2.4, 3.2},
        {0.5, 0.7, 1.2},
        {2.2, 2.9, 3.1},
        {1.9, 2.2, 2.9},
        {3.1, 3.0, 3.3},
    };*/
    std::vector<std::vector<double>> testData = {
        {1,2},
        {-1,-2}
    };
    int k = 1;
	PCA p;
    std::vector<std::vector<double>> reducedData = p.transform(testData, k);

    std::cout << "Reduced Data:" << std::endl;
    for (const auto& row : reducedData) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}



/*
int main() {
	Eigen::MatrixXf mat_1(6, 4);
	Eigen::MatrixXf mat_2(4, 6);
	matrix cov_1 = matrix::Zero(6, 6);
	matrix cov_2 = matrix::Zero(6, 6);

	mat_2 <<	0.43,	1.1,	0,	0.32,	0.21,	0,
				0.4,	1.28,	0,	0.34,	0.17,	0,
				0.39,	1.2,	0,	0.31,	0.25,	0,
				0.45,	1.4,	0,	0.22,	0.13,	0;
	mat_1 = mat_2.transpose();

	//1 各个维度的均值
	matrix mean_1 = mat_1.rowwise().mean();
	matrix mean_2 = mat_2.colwise().mean();

	//2 每个样本减去均值
	for (size_t i = 0; i < mat_1.cols(); ++i)
	{
		cov_1.col(i) = mat_1.col(i) - mean_1;
	}
	for (size_t i = 0; i < mat_2.rows(); ++i) {
		cov_2.row(i) = mat_2.row(i) - mean_2;
	}
	//3 计算协方差
	cov_1 = cov_1 * cov_1.transpose() / (cov_1.cols() - 1);
	cov_2 = cov_2.transpose() * cov_2 / (cov_2.rows() - 1);
	std::cout << cov_1 << std::endl << std::endl << cov_2;

	return 0;
}
*/