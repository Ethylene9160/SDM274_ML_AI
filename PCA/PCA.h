#ifndef PAC_H_ETHY
#define PAC_H_ETHY 1
#include<bits/stdc++.h>
#include<math.h>
#include<Eigen/SVD>
#include<Eigen/Dense>
#include<cassert>
#ifndef ETHY_DOUBLE_VECTOR


typedef std::vector<double> my_vector;
#endif
#include <Eigen/Dense>
#include <vector>

class PCA {
public:
    PCA(int k);
    PCA();

    void train(const std::vector<std::vector<double>>& data);

    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& data, int k);

    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& data);

private:
    int k;
    bool hasTrained;
    Eigen::MatrixXd transformedMatrix;
};
/*
class PCA {
private:
    Eigen::MatrixXd transformed;
	double mean(my_vector& x);

	double var(my_vector& x);

	double cov_var(my_vector& x, my_vector& y) {
		int size = x.size(), sizey = y.size();
		assert(size == sizey);
		double xpi = 0.0, ypi = 0.0, cov_res = 0.0;
		for (int i = 0; i < size; ++i) {
			xpi += x[i];
			ypi += y[i];
		}
		xpi /= double(size);
		ypi /= double(size);

		for (int i = 0; i < size; ++i) {
			cov_res += (x[i]-xpi) * (y[i]-ypi);
		}
		cov_res /= double(size);
		return cov_res;
	}

public:
    std::vector<std::vector<double>> pca(const std::vector<std::vector<double>>& data, int k) {
        assert(data.size() > 0);
        assert(k <= data[0].size());
        // 将std::vector转换为Eigen::MatrixXd
        int rows = data.size();
        int cols = data[0].size();
        Eigen::MatrixXd dataMat(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                dataMat(i, j) = data[i][j];
            }
        }

        // 计算均值并中心化数据
        Eigen::VectorXd mean = dataMat.colwise().mean();
        Eigen::MatrixXd centered = dataMat.rowwise() - mean.transpose();

        // 计算协方差矩阵
        Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(dataMat.rows() - 1);

        // 计算特征值和特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
        Eigen::MatrixXd eigVectors = eig.eigenvectors().rightCols(k);

        // 转换到新的子空间
        this->transformed = centered * eigVectors;

        // 将结果转换回std::vector
        std::vector<std::vector<double>> result;
        result.reserve(this->transformed.rows());
        for (int i = 0; i < this->transformed.rows(); ++i) {
            std::vector<double> row;
            row.reserve(k);
            for (int j = 0; j < k; ++j) {
                row.push_back(transformed(i, j));
            }
            result.push_back(row);
        }

        return result;
    }

    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& data, int k) {
        

        
    }
};
*/
#endif
