#ifndef MY_SVM_H
#define MT_SVM_H 1
#include<bits/stdc++.h>
#include<Eigen/Dense>

typedef std::vector<double> my_vector;
typedef std::vector<std::vector<double>> my_matrix;

Eigen::MatrixXf array2matrix(std::vector<std::vector<double>>& input) {
	int row = input.size();
	int col = input[0].size();
	Eigen::MatrixXf output(row, col);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++k) {
			output(i, j) = input[i][j];
		}
	}
	return output;
}

Eigen::MatrixXf array2matrix(std::vector<double>& input) {
	int col = input.size();
	Eigen::MatrixXf output(1, col);
	for (int i = 0; i < col; ++i) {
		output(1, i) = input[i];
	}
	return output;
}

std::vector<std::vector<double>> matrix2array(Eigen::MatrixXf& input) {
	int row = 0;
	int col = 0;
	std::vector<std::vector<double>> output(row, std::vector<double>(col));
	//todo
	return output;
}

double mult(my_vector& x, my_vector& y) {
	int size = x.size();
	assert(y.size() == size);
	double out = 0;
	for (int i = 0; i < size; ++i) {
		out += x[i] * y[i];
	}
	return out;
}


class kernal {
public:
	virtual double kernalFunction(double x, double y);
};

class svm {
protected:
	kernal k;
	Eigen::MatrixXf W;	//power matrix;
	Eigen::MatrixXf b;	//bias matrix;
	my_vector alpha;
	
public:
	void fit(std::vector<std::vector<double>>&X, std::vector<double>&y) {
		int data_length = X.size();
		assert(data_length == y.size());
		auto X_mat = array2matrix(X);
		auto y_mat = array2matrix(y);
		fit(X_mat, y_mat);
	}

	void fit(Eigen::MatrixXf& X, Eigen::MatrixXf& y) {

	}

	int predict(std::vector<double>& X) {
		auto input_mat = array2matrix(X);
		return this->W.dot(X) + b > 0;
	}
};



#endif
