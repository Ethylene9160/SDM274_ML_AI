#ifndef MY_SVM
#define MY_SVM 1
#include<bits/stdc++.h>
#include<Eigen/Dense>
//#include"libsvm/svm.h"

typedef std::vector<double> my_vector;
typedef std::vector<std::vector<double>> my_matrix;
typedef Eigen::MatrixXd MyMatrix;

static MyMatrix array2matrix(std::vector<std::vector<double>>& input) {
	int row = input.size();
	int col = input[0].size();
	MyMatrix output(row, col);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			output(i, j) = input[i][j];
		}
	}
	return output;
}

static Eigen::VectorXd array2Vector(std::vector<double>& input) {
	Eigen::VectorXd output(input.size());
	for (int i = 0; i < input.size(); ++i) {
		output(i) = input[i];
	}
	return output;
}

static std::vector<std::vector<double>> matrix2array(MyMatrix& input) {
	int row = 0;
	int col = 0;
	std::vector<std::vector<double>> output(row, std::vector<double>(col));
	//todo
	return output;
}

static double mult(my_vector& x, my_vector& y);

static MyMatrix computeKernelMatrix(const MyMatrix& X, const MyMatrix&X2) {
	int n = X.rows();
	//MyMatrix K(n, n);
	return X * X2;
	/*
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			K(i, j) = X.row(i).dot(X.row(j));
		}
	}
	*/
	//return K;
}

static Eigen::VectorXd conputeKernalMatrix(const MyMatrix& X, const Eigen::VectorXd& X2) {
	return X * X2.transpose();
}

static Eigen::VectorXd calculateLinearKernal(const MyMatrix& X, const Eigen::VectorXd& X2) {
	return X * X2;
}

static MyMatrix calculateLinearKernal(const MyMatrix& X, const MyMatrix& X2) {
	return X * X2.transpose();
}

static double default_kernal(my_vector&input, int i) {
	return input[i];//pass. TODO!
}

static int randomInt(int i, int n) {
	int j = i;
	while (j == i) {
		j = rand() % n;
	}
	return j;
}

template<typename T>
static T max(T a, T b) {
	return a > b ? a : b;
}

template<typename T>
static T min(T a, T b) {
	return a < b ? a : b;
}



/*
class PY_SVM {
	svm s;
public:
	void init(double learning_rate) {
		s.setLearningRate(learning_rate);
	}

	void train(std::vector<std::vector<double>>& X, std::vector<int>& y) {
		s.fit(X, y);
	}


};*/

class svm {
protected:
	//calculateLinearKernal k;
	MyMatrix W;	//power matrix;
	MyMatrix B;	//bias matrix;
	my_matrix alpha;
	Eigen::VectorXd alpha_mat;
	bool hasTrained;
	//double (*calculateLinearKernal)(my_vector&input, int i);
	MyMatrix (*kernal0)(const MyMatrix& X, const MyMatrix&Y);
	Eigen::VectorXd (*kernal)(const MyMatrix& X, const Eigen::VectorXd& Y);
	Eigen::VectorXd w;
	double b;
	double learning_rate;

	double C;

	int epochs;
	/*
	Eigen::VectorXd calculateGradient(const MyMatrix& K, const std::vector<int>& labels, const Eigen::VectorXd& alpha) {
		int n = alpha.size();
		Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(labels.data(), labels.size());
		Eigen::VectorXd sum = K * (alpha.cwiseProduct(y));
		return Eigen::VectorXd::Ones(n) - y.cwiseProduct(sum);
	}*/

	
	Eigen::VectorXd calculateGradient(const MyMatrix& X, const std::vector<int>& labels, const Eigen::VectorXd& alpha) {
		//printf("calculate gradient\n");
		int n = X.rows(); // 数据点的数量
		Eigen::VectorXd gradient(n);

		for (int k = 0; k < n; ++k) {
			double sum = 0.0;
			for (int j = 0; j < n; ++j) {
				// 计算 x_k 和 x_j 的点积
				double dot_product = X.row(k).dot(X.row(j));
				// 累加 y_k * y_j * alpha_j * x_k * x_j
				sum += labels[k] * labels[j] * alpha[j] * dot_product;
			}
			// 计算梯度的第 k 个分量
			gradient[k] = 1 - sum;
		}

		return gradient;
	}

	double clip(double a, double L, double H) {
		if (a > H) return H;
		if (a < L) return L;
		return a;
	}

	void findBound(double& L, double& H, Eigen::VectorXd&a, int i, int j, std::vector<int> const&labels) {
		if (labels[i] == labels [j] ) {
			L = max(0.0, double(a(i)+a(j)-this->C));
			H = min(double(a(i) + a(j)), this->C);
		}
		else {
			//知乎提供的原始代码在此处存在错误
			L = max(0.0, double(a(j) - a(i)));
			H = min(this->C, double(this->C - a(i) + a(j)));
		}
	}

public:
	svm() {
		this->kernal = calculateLinearKernal;
		//this->kernal = computeKernelMatrix;
		this->C = 0.6;
		printf("init successfully!\n");
	}

	void fit(std::vector<std::vector<double>>&X, std::vector<int>&y) {
		printf("start fit\n");
		int data_length = X.size();
		//assert(data_length == y.size());
		MyMatrix X_mat = array2matrix(X);
		//auto y_mat = array2matrix(y);
		fit(X_mat, y);
	}

	//Deprecated
	void fit(MyMatrix& X, MyMatrix& y) {
		int row = 0; //todo
		int col = 0; //todo
		this->alpha = std::vector<std::vector<double>>(row, std::vector<double>(col,0.0));
		for (int i = 0; i < row; ++i) {

		}

		//todo: update alpha!!!
		this->W = MyMatrix(row, col);



		hasTrained = 1;
	}

	double computeBias(const MyMatrix& X, const std::vector<int>& labels, const Eigen::VectorXd& alpha) {
		double b = 0.0;
		int supportVectorCount = 0;
		for (int i = 0; i < alpha.size(); ++i) {
			if (alpha[i] > 0) { // 如果是支持向量
				double sum = 0.0;
				for (int j = 0; j < alpha.size(); ++j) {
					sum += alpha[j] * labels[j] * X.row(j).dot(X.row(i));
				}
				b += (labels[i] - sum);
				supportVectorCount++;
			}
		}
		return supportVectorCount > 0 ? b / supportVectorCount : 0;
	}

	void fit(MyMatrix const & X, std::vector<int> const& labels) {

		int n = X.rows();
		Eigen::VectorXd a(n);//todo: check whether the init is corrrect.
		Eigen::VectorXd y(n);
		for (uint32_t i = 0; i < n; ++i) {
			a[i] = 0.0;
			y[i] = double(labels[i]);
		}



		double b = 0.0;
		this->epochs = 1; // 最大迭代次数
		MyMatrix K = calculateLinearKernal(X,X);
		printf("init over\n");
		for (uint32_t iter = 0; iter < this->epochs; ++iter) {
			printf("%dth epoch\n", iter);
			//Eigen::VectorXd a_prev(a);// shallow copy
			Eigen::VectorXd& a_prev = a;// todo: adjust this.
			for (uint32_t i = 0; i < n; ++i) {
				printf("i=%d\n", i);
				std::cout << X << std::endl;
				auto temp = Eigen::VectorXd(X.row(i));
				std::cout << temp << std::endl;
				Eigen::VectorXd Ki = kernal(X, temp);
				//checkpoint: dimension of which.
				printf("finished K%d\n", i);
				std::cout << Ki << std::endl;
				double ui = (a_prev(i)*y).dot(Ki) + b;
				double Ei = ui - y(i);
				printf("judge y ei ai c\n");
				if ((y[i] * Ei >= 1.0 && a_prev[i] > 0) ||
					(y[i] * Ei<=1 && a_prev[i] < this->C)||
					(y[i] = Ei == 1 && a_prev[i] == 0) ||
					(y[i] * Ei == 1 && a_prev[i] == this->C) ){
					printf("inner judgement\n");
					int j = randomInt(i, n);
					// error for i
					Eigen::VectorXd Kj = calculateLinearKernal(X, Eigen::VectorXd(X.row(j)));
					double uj = (a(j) * y).dot(Kj) + b;
					double Ej = uj - y[j];
					//find bounds
					double L = 0, H = 0;
					this->findBound(L, H, a, i, j, labels);
					printf("L: %f, H: %f\n", L, H);
					//eta
					double eta = K(i, i) + K(j, j) - 2 * K(i, j);
					static double epsilon = 1e-6;
					if (eta <= epsilon) continue;
					
					//save old alphas 
					double ai_old = a_prev[i];
					double aj_old = a_prev[j];

					//update
					a[j] = aj_old + y[j] * (Ei - Ej) / eta;
					a[j] = this->clip(a[j], L, H);
					a[i] = ai_old + y[j] / y[i] * (aj_old - a_prev[j]);
					
					// find intercept
					double b1 = b - Ei - y[i] * (a[i] - ai_old) * K(i, i) - y[j] * (a[j] - aj_old) * K(i, j);
					double b2 = b - Ei - y[i] * (a[i] - ai_old) * K(i, j) - y[j] * (a[j] - aj_old) * K(j, j);

					if (0.0 < a[i] && a[i] < this->C) {
						b = b1;
					}
					else if (0.0 < a[j] && a[j] < this->C) {
						b = b2;
					}
					else {
						b = 0.5 * (b1 + b2);
					}
				}
				else {
					continue;
				}
			}
			this->alpha_mat = a;
			//auto diff = sqrt()
			std::cout << "a is:\n" << a << std::endl;

		}




		printf("end fit\n");
		//return;
		// Generated by chatGPT
		//Eigen::VectorXd alpha; // 初始化 \(\alpha\)
		//this->alpha.setZero(X.rows());
		//this->alpha_mat.setZero(X.rows());
		this->learning_rate = 0.006; // 学习率
		
		this->w = Eigen::VectorXd(X.cols());
		for (int i = 0; i < w.size(); ++i) {
			w[i] = 0.0;
		}
		std::cout << "[alpha is]:\n";
		std::cout << alpha_mat << std::endl;
		/*
		for (int iter = 0; iter < epochs; ++iter) {
			Eigen::VectorXd gradient = calculateGradient(X, labels, alpha_mat);
			// 更新 \(\alpha\)
			alpha_mat = alpha_mat - learning_rate * gradient;
			//this->b = b + learning_rate * gradient;
			// 可以添加收敛条件检查
		}
		this->b = computeBias(X, labels, alpha_mat);
		*/
		printf("start calculate w\n");
		printf("size of alpha_mat is:%d\n", alpha_mat.rows());
		printf("size of labels is:%d\n", labels.size());
		printf("size of X's row and col are: %d, %d\n", X.rows(), X.cols());
		printf("size of w is:%d\n", w.size());
		for (int i = 0; i < X.cols(); ++i) {
			Eigen::VectorXd x = X.col(i);
			for (int j = 0; j < X.rows(); ++j) {
				this->w(i) += this->alpha_mat(j) * (double)labels[j];
			}
		}
		printf("finish calculate w\n");
	}
	/**
	* Above the line, then return 1; else return 0.
	*/
	/*
	int predict(std::vector<double>& input) {
		int size = input.size();
		std::vector<double> X = input;
		for (int i = 0; i < size; ++i) {
			//todo.
			X[i] = calculateLinearKernal(input, i);
		}
		assert(hasTrained);
		auto input_mat = array2matrix(X);
		auto output_mat = this->W* input_mat + B;
		return output_mat(0, 0) > 0;
	}*/

	std::vector<int> predict(std::vector<std::vector<double>>& X) {
		printf("predict\n");
		std::vector<int> predictions;
		predictions.reserve(X.size());
		for (int i = 0; i < this->w.size(); ++i) {
			printf("w%d is: %f\t", i, w(i));
		}
		printf("\n");
		printf("b is:%f\n", this->b);
		for (const auto& x_vec : X) {
			Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(x_vec.data(), x_vec.size());

			double decisionValue = this->w.dot(x);
			int predictedLabel = decisionValue >= 0 ? 1 : -1;
			predictions.push_back(predictedLabel);
		}

		return predictions;
	}

	void setLearningRate(double lr) {
		this->learning_rate = lr;
	}
};

#endif
