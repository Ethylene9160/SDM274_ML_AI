#include "PCA.h"

PCA::PCA(int k) : k(k), hasTrained(false) {}

PCA::PCA() : PCA(1) {}

void PCA::train(const std::vector<std::vector<double>>& data) {
    // [��֮ǰ��ͬ�Ĳ��裬���ڼ���ת������]
    assert(data.size() > 0);

    // ��std::vectorת��ΪEigen::MatrixXd
    int rows = data.size();
    int cols = data[0].size();
    Eigen::MatrixXd dataMat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dataMat(i, j) = data[i][j];
        }
    }

    // �����ֵ�����Ļ�����
    Eigen::VectorXd mean = dataMat.colwise().mean();
    Eigen::MatrixXd centered = dataMat.rowwise() - mean.transpose();

    // ����Э�������
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(dataMat.rows() - 1);

    // ��������ֵ����������
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

    // [��dataת��ΪEigen���󲢽������Ļ�]
    assert(data.size() > 0);
    assert(data.size() >= k);
    // ��std::vectorת��ΪEigen::MatrixXd
    int rows = data.size();
    int cols = data[0].size();
    Eigen::MatrixXd dataMat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dataMat(i, j) = data[i][j];
        }
    }
    // �����ֵ�����Ļ�����
    Eigen::VectorXd mean = dataMat.colwise().mean();
    Eigen::MatrixXd centered = dataMat.rowwise() - mean.transpose();
    // ʹ�ñ����ת��������н�ά
    Eigen::MatrixXd transformed = centered * transformedMatrix;

    // [�����ת����std::vector������]
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
