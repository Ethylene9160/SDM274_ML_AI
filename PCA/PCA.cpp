#include "PCA.h"

PCA::PCA(int k) : k(k), hasTrained(false) {}

PCA::PCA() : PCA(1) {}

void PCA::train(const std::vector<std::vector<double>>& data) {
    // [与之前相同的步骤，用于计算转换矩阵]
    assert(data.size() > 0);

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
    this->transformedMatrix = eig.eigenvectors().rightCols(k);

    hasTrained = true;
    printf("train over.\n");
}

std::vector<std::vector<double>> PCA::transform(const std::vector<std::vector<double>>& data, int k) {
    if (!hasTrained) {
        //throw std::runtime_error("PCA has not been trained");
        printf("untrained! it's going to train...\n");
        train(data);
    }

    // [将data转换为Eigen矩阵并进行中心化]
    assert(data.size() > 0);
    assert(data.size() >= k);
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
    // 使用保存的转换矩阵进行降维
    Eigen::MatrixXd transformed = centered * transformedMatrix;

    // [将结果转换回std::vector并返回]
    std::vector<std::vector<double>> result;
    result.reserve(transformed.rows());
    for (int i = 0; i < transformed.rows(); ++i) {
        std::vector<double> row;
        row.reserve(k);
        for (int j = 0; j < k; ++j) {
            row.push_back(transformed(i, j));
        }
        result.push_back(row);
    }

    return result;
}

std::vector<std::vector<double>> PCA::transform(const std::vector<std::vector<double>>& data) {
    return this->transform(data, this->k);
}
