#ifndef _ETHY_NEURON_
#define _ETHY_NEURON_ 1

//#include<bits/stdc++.h>
#include<vector>
#include<iostream>
#include <cassert>
#include <stdexcept> // 用于抛出异常
#include <random> //随机数生成
#include <ctime>

typedef std::vector<double> my_vector;// to make it easier for programming.
typedef std::vector<my_vector> single_power;
typedef std::vector<single_power> my_power;

/*
the power between layer (h, i) and layer (h+1, j) will be stored in:
w[h][i][j]. w is an instance of my_power(3d-vector of double)
*/

/*
* 为单个输出和预测值计算MSE损失。
calculate MSE loss for single output and single prediction.
*/
/*
* 为单个输出和预测值计算MSE损失。
calculate MSE loss for single output and single prediction.
*/
inline double getMSELoss(double x1, double x2) {
	double d = x1 - x2;
	return d * d;
}

class LinearAutoEncoder {
protected:
	int LR_VOKE;					//after LR_VOKE epoches, the prog will print the loss value , current w and b.
	int epoches;					//learning times
	int middle = 1;
	double learning;				//study rate
	//double lambda;					//batch normalization
	my_power w;						//power, dimension is three
	//my_vector output;				//dinal output, dimension is one
	std::vector<my_vector> h;		//layer output storage; dimension is 2;
	std::vector<my_vector> batch_o;	//using batch normalization.
	std::vector<my_vector> o;		//after sigmoid output layer.
	std::vector<my_vector> b;		//bias, dimension is 2 to fix each layer h.

	void init(int epoches, double lr, std::vector<my_vector> h)
	{
		this->LR_VOKE = 500;
		this->epoches = epoches;
		this->learning = lr;
		//this->middle = 1;
		//this->layers = layers;
		this->h = h;
		this->batch_o = h;
		this->o = h;
		this->b = std::vector<my_vector>(h.size());
		
		//printf("size of h is: \n");
		for (int i = 0; i < h.size(); ++i) {
			//printf("size%d is %d\n", i, h[i].size());
		}
		//printf("\n");
		//printf("size of b is: \n");
		for (int i = 0; i < h.size(); ++i) {
			b[i] = my_vector(h[i].size(), 0);
			//printf("size%d is %d\n", i, b[i].size());
		}


		this->w = my_power(this->h.size() - 1);
		for (int i = 0; i < this->w.size(); ++i) {
			this->w[i] = single_power(h[i].size(), my_vector(h[i + 1].size()));
		}
		// 随机赋予w值
		// 初始化随机数生成器
		std::default_random_engine generator(static_cast<unsigned>(time(0)));
		std::uniform_real_distribution<double> distribution(-1.0, 1.0); // 例如在-1.0到1.0之间生成随机数

		// 初始化权重矩阵
		this->w = my_power(this->h.size() - 1);
		for (int i = 0; i < this->w.size(); ++i) {
			this->w[i] = single_power(h[i].size(), my_vector(h[i + 1].size()));
			for (int j = 0; j < w[i].size(); ++j) {
				for (int k = 0; k < w[i][j].size(); ++k) {
					w[i][j][k] = distribution(generator); // 使用随机数填充
				}
				b[i][j] = distribution(generator);
			}
		}


		//printf("\nsize of w is: \n");
		for (int i = 0; i < w.size(); ++i) {
			//printf("size of %dth of w is %d * %d\n", i, w[i].size(), w[i][0].size());
		}

		//printf("================\n init w will be: \n");
		for (int i = 0; i < w.size(); ++i) {
			//printf("w%d will be:\n", i);
			for (int j = 0; j < w[i].size(); ++j) {
				for (int k = 0; k < w[i][j].size(); ++k) {
					//printf("%f\t", w[i][j][k]);
				}
				//printf("\n");
			}
		}
		//printf("================\n init b will be: \n");
		for (int i = 0; i < b.size(); ++i) {
			for (int j = 0; j < b[i].size(); ++j) {
				//printf("b%d%d is: %f\t", i, j, b[i][j]);
			}
			//printf("\n");
		}

	}

	bool isSameDouble(double d1, double d2)
	{
		return d1 == d2;
	}

	void calculateOutput(my_vector& x, my_vector& y, my_vector& patch_output, single_power& power, my_vector& bias, my_vector& o_sigmoid, bool shouldOrth)
	{
		// 确保权重矩阵的列数与输出向量的尺寸相匹配
		//for (int i = 0; i < power.size(); ++i) {
		//assert(power[i].size() == y.size());
		//}

		// 确保偏置向量的尺寸与输出向量的尺寸相匹配
		//assert(bias.size() == y.size());
		//my_vector x = x0;

		orth(x, x);
		// 计算输出向量的每个元素
		int y_size = y.size();
		for (int j = 0; j < y_size; ++j) {
			y[j] = 0.0;
			for (int k = 0; k < x.size(); k++) {
				// 确保权重矩阵的行数与输入向量的尺寸相匹配
				//assert(k < power.size());
				// 确保当前的k行有一个对应的j列
				//assert(j < power[k].size());
				y[j] += x[k] * power[k][j];
			}
			//更新y
			y[j] = (y[j] + bias[j]);
			o_sigmoid[j] = this->sigmoid(y[j]);
		}

		//if(shouldOrth) 
		//orth(y, y);
		//return;
		//for (int i = 0; i < y_size; ++i) {
		//o_sigmoid[i] = this->sigmoid(y[i]);
		//}





		/*
		for (int j = 0; j < y.size(); ++j) {
		y[j] = 0.0;
		for (int k = 0; k < x.size(); k++) {
		y[j] += x[k] * power[k][j];
		}
		y[j] = this->sigmoid(y[j]+bias[j]);
		}*/
	}

	virtual void orth(my_vector& self_h, my_vector& batch_output)
	{




	}

public:
	/*
	@ params:
	epoches: 迭代次数
	lr: 学习率
	h: 包含了输入、输出层的所有layer。取输出时，仅取h.back()[0]作为输出参考。
	@ Deprecated
	*/
	LinearAutoEncoder(int epoches, double lr, std::vector<my_vector> h)
	{
		this->init(epoches, lr, h);
	}
	//MyNeuron(int epoches, double lr, std::vector<std::)

	/*
	@ params:
	epoches: 迭代次数
	lr: 学习率
	inputSize: 输入维度
	hiddenLayerSizes: 这个数组的的长度将会是隐藏层的长度，这个数组中的每个元素将会是每个隐藏层的维度。
		例如输入{2,3,3}, 隐藏层第一层神经元数量是2，第二、三层神经元数量都是3.

	*/
	LinearAutoEncoder(int epoches, double lr, int inputSize, std::vector<int> hiddenLayerSizes)
	{
		int size = hiddenLayerSizes.size();
		std::vector<my_vector> v(size + 1);
		v[0] = my_vector(inputSize);

		for (int i = 0; i < size; ++i) {
			v[i + 1] = my_vector(hiddenLayerSizes[i]);
		}
		v.push_back(v[0]);
		this->middle = hiddenLayerSizes.size()/2 + 1;
		init(epoches, lr, v);
		printf("middle is : %d\n", middle);
	}

	/*
	@ params:
	epoches: 迭代次数
	lr: 学习率
	inputSize: 输入维度
	hiddenLayerSizes: 这个数组的的长度将会是隐藏层的长度，这个数组中的每个元素将会是每个隐藏层的维度。
		例如输入{2,3,3}, 隐藏层第一层神经元数量是2，第二、三层神经元数量都是3.
	hidenSize: 由于参数传递是一个数组，所以需要传入一个数组（隐藏层数量）的长度。
	@ comments
	这个函数主要用于适配python，用于防止C++中vector传递和python中元组的传递出现的问题。
	*/
	LinearAutoEncoder(int epoches, double lr, int inputSize, int hiddenLayerSizes[], int hidenSize)
	{
		//printf("test start\n");
		std::vector<my_vector> v(hidenSize + 1);
		v[0] = my_vector(inputSize);
		//printf("input over\n");
		for (int i = 0; i < hidenSize; ++i) {
			v[i + 1] = my_vector(hiddenLayerSizes[i]);
		}

		v.push_back({ { 0.0 } });
		//printf("start init!!\n");
		init(epoches, lr, v);
	}



	/*
	激活函数。
	*/
	virtual double sigmoid(double x)
	{
		//return x;
		//return x > 0 ? x : 0.01*x;
		//return x;
		return 1 / (1 + exp(-x));
	}

	/*
	激活函数的导函数
	*/
	virtual double d_sigmoid(double x)
	{
		//return 1;
		// the derivaty of 1/(1+exp(-x)) will be:
		// exp(-x)(1+exp(-x))^-2
		// =1/(1+exp(-x)) times exp(-x)/(1+exp(-x))
		// to lower the time complexity, use y.
		//return x > 0 ? 1.0 : 0.01;
		double y = sigmoid(x);
		return y * (1 - y);
	}



	my_vector& forward(my_vector& data) {
		// 确保输入数据的尺寸与网络输入层的尺寸相匹配
		//if (data.size() != h[0].size()) {
		//throw std::invalid_argument("Size of input data does not match the size of the network's input layer.");
		//}

		// 用输入数据初始化第一层的输出
		h[0].assign(data.begin(), data.end());
		o[0].assign(data.begin(), data.end());//输入层，不需要sigmoid
		// 进行前向传播
		int i_max = this->h.size() - 1;
		for (int i = 0; i < i_max; i++) {
			// 在进行矩阵乘法之前，确保索引有效
			//if (i >= w.size() || i + 1 >= h.size() || i + 1 >= b.size()) {
			//throw std::out_of_range("Index out of range during forward pass.");
			//}

			// 检查权重矩阵的维度是否正确
			//if (h[i].size() != w[i].size()) {
			//    throw std::invalid_argument("Mismatch between layer output size and weight matrix size.");
			//}

			// 检查权重矩阵的每个向量的尺寸是否与下一层的尺寸匹配
			//for (int k = 0; k < w[i].size(); ++k) {
			//if (w[i][k].size() != h[i + 1].size()) {
			//throw std::invalid_argument("Mismatch between weight matrix size and next layer size.");
			//}
			//}

			// 计算下一层的输出
			//this->calculateOutput(h[i], h[i + 1], w[i], b[i + 1]);
			this->calculateOutput(o[i], h[i + 1], batch_o[i + 1], w[i], b[i + 1], o[i + 1], i + 1 < i_max);
			//printf("new h will be:\n");
			//for (int ti = 0; ti < h[i+1].size(); ++ti) {

			//printf("h%d%d is: %f\t", i+1, ti, h[i+1][ti]);
			//}
		}

		// 返回最后一层的输出
		//return this->h[this->h.size() - 1];
		return this->o[this->o.size() - 1];
	}
	//my_vector forward(std::vector<my_vector>& data);

	/*
	训练。计算权重w和偏置b。
	该函数不会对data和label进行写入操作。
	@ params
	data: 二维（m*n）训练数组。m表示样本数量， n表示每个输入维数的。
		需要注意的是，输入维数n需要和layer的第一层（即h[0]）的维度相同。否则程序会退出。
	label：测试样本的参考输出。是一个一维的长度为m的double数组。
	*/
	void train(std::vector<my_vector>& data, my_vector& label) {
		//assert(data.size() == label.size());  // 确保数据和标签的数量匹配
		for (int epoch = 0; epoch < epoches; ++epoch) {
			for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
				//assert(dataIndex < label.size());  // 确保标签索引在范围内

				//printf("train-forward\n");
				// 前向传播
				my_vector output = forward(data[dataIndex]);
				my_vector& output_h = this->h[h.size() - 1];

				//assert(!output.empty());  // 确保输出不为空
				//printf("train-gradient\n");
				// 计算输出层的梯度
				my_vector outputLayerGradient;

				for (int neuronIndex = 0; neuronIndex < 1; ++neuronIndex) {
					double error = label[dataIndex] - output[neuronIndex];
					//outputLayerGradient.push_back(error * d_sigmoid(output[neuronIndex]));
					//if (error >= 0) error = 0;
					outputLayerGradient.push_back(error * d_sigmoid(output_h[neuronIndex]));
				}
				//printf("train-backward\n");
				// 反向传播
				std::vector<my_vector> layerGradients;
				layerGradients.push_back(outputLayerGradient);
				for (int layerIndex = h.size() - 2; layerIndex >= 0; --layerIndex) {
					//assert(layerIndex < w.size());  // 确保权重索引在范围内
					my_vector layerGradient;
					for (int neuronIndex = 0; neuronIndex < h[layerIndex].size(); ++neuronIndex) {
						double gradientSum = 0;
						for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < h[layerIndex + 1].size(); ++nextLayerNeuronIndex) {
							//assert(layerIndex < w.size() && neuronIndex < w[layerIndex].size() && nextLayerNeuronIndex < w[layerIndex][neuronIndex].size()); // 确保权重索引在范围内
							gradientSum += w[layerIndex][neuronIndex][nextLayerNeuronIndex] * layerGradients.back()[nextLayerNeuronIndex];
							//if (gradientSum >= 0) gradientSum = 0;//梯度剪切
						}
						layerGradient.push_back(gradientSum * d_sigmoid(h[layerIndex][neuronIndex]));
					}
					layerGradients.push_back(layerGradient);
				}
				//printf("train-re-new\n");
				// 更新权重和偏置
				for (int layerIndex = 0; layerIndex < w.size(); ++layerIndex) {
					for (int neuronIndex = 0; neuronIndex < w[layerIndex].size(); ++neuronIndex) {
						for (int nextNeuronIndex = 0; nextNeuronIndex < w[layerIndex][neuronIndex].size(); ++nextNeuronIndex) {
							//assert(layerIndex < h.size() && neuronIndex < h[layerIndex].size());  // 确保 h 的索引在范围内
							//assert(neuronIndex < w[layerIndex].size() && nextNeuronIndex < w[layerIndex][neuronIndex].size());
							//w[layerIndex][neuronIndex][nextNeuronIndex] += learning * h[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
							w[layerIndex][neuronIndex][nextNeuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
							//printf("w is good!\n");
						}
						//b[layerIndex][neuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
						//printf("finished cal w%d%d\n", layerIndex, neuronIndex);
					}
					//printf("finish cal w%d\n", layerIndex);
					//assert(layerIndex < b.size());  // 确保偏置索引在范围内
					for (int biasIndex = 0; biasIndex < b[layerIndex].size(); ++biasIndex) {
						b[layerIndex][biasIndex] += learning * layerGradients[layerGradients.size() - 1 - layerIndex][biasIndex];
						//assert(layerIndex < layerGradients.size() && layerIndex >= 0);
						//if (biasIndex >= layerGradients[layerIndex].size())
						//  printf("bias index(%d) is larger than the size(%d)!\n", biasIndex, layerGradients[layerIndex].size());
						//assert(biasIndex < layerGradients[layerIndex].size());

						//b[layerIndex][biasIndex] += learning * layerGradients[layerIndex][biasIndex];
					}
					//printf("finish cal b%d\n", layerIndex);

				}
				//printf("finish cal w\n");
			}
			// 每个epoch后输出损失
			//continue;
			if (epoch % LR_VOKE) continue;
			//printf("train-printloss\n");
			double loss = 0;
			for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
				my_vector& output = forward(data[dataIndex]);
				for (int outputIndex = 0; outputIndex < output.size(); ++outputIndex) {
					//double error = label[dataIndex] - output[outputIndex];
					//loss += error * error;  // MSE
					loss += getMSELoss(label[dataIndex], output[outputIndex]);
				}
			}
			loss /= data.size();
			std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
			/*
			for (int i = 0; i < w.size(); ++i) {
			for (int j = 0; j < w[i].size(); ++j) {
			printf("w%d%d:\t", i, j);
			for (int k = 0; k < w[i][j].size(); ++k) {
			printf("%f\t", w[i][j][k]);
			}
			printf("\nb: %f\n", b[i][j]);
			}
			printf("\n");
			}*/

		}

		double loss = 0;
		for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
			my_vector& output = forward(data[dataIndex]);
			for (int outputIndex = 0; outputIndex < output.size(); ++outputIndex) {
				//double error = label[dataIndex] - output[outputIndex];
				//loss += error * error;  // MSE
				loss += getMSELoss(label[dataIndex], output[outputIndex]);
			}
		}
		loss /= data.size();
		std::cout << "Loss: " << loss << std::endl;
	}

	void train(std::vector<my_vector>& data, std::vector<my_vector>& label) {
		//assert(data.size() == label.size());  // 确保数据和标签的数量匹配
		for (int epoch = 0; epoch < epoches; ++epoch) {
			for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
				//assert(dataIndex < label.size());  // 确保标签索引在范围内

				//printf("train-forward\n");
				// 前向传播
				my_vector output = forward(data[dataIndex]);
				my_vector& output_h = this->h[h.size() - 1];

				//assert(!output.empty());  // 确保输出不为空
				//printf("train-gradient\n");
				// 计算输出层的梯度
				my_vector outputLayerGradient;

				for (int neuronIndex = 0; neuronIndex < output_h.size(); ++neuronIndex) {
					double error = label[dataIndex][neuronIndex] - output[neuronIndex];
					//outputLayerGradient.push_back(error * d_sigmoid(output[neuronIndex]));
					outputLayerGradient.push_back(error * d_sigmoid(output_h[neuronIndex]));
				}
				//printf("train-backward\n");
				// 反向传播
				std::vector<my_vector> layerGradients;
				layerGradients.push_back(outputLayerGradient);
				for (int layerIndex = h.size() - 2; layerIndex >= 0; --layerIndex) {
					//assert(layerIndex < w.size());  // 确保权重索引在范围内
					my_vector layerGradient;
					for (int neuronIndex = 0; neuronIndex < h[layerIndex].size(); ++neuronIndex) {
						double gradientSum = 0;
						for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < h[layerIndex + 1].size(); ++nextLayerNeuronIndex) {
							//assert(layerIndex < w.size() && neuronIndex < w[layerIndex].size() && nextLayerNeuronIndex < w[layerIndex][neuronIndex].size()); // 确保权重索引在范围内
							gradientSum += w[layerIndex][neuronIndex][nextLayerNeuronIndex] * layerGradients.back()[nextLayerNeuronIndex];
						}
						layerGradient.push_back(gradientSum * d_sigmoid(h[layerIndex][neuronIndex]));
					}
					layerGradients.push_back(layerGradient);
				}
				//printf("train-re-new\n");
				// 更新权重和偏置
				for (int layerIndex = 0; layerIndex < w.size(); ++layerIndex) {
					for (int neuronIndex = 0; neuronIndex < w[layerIndex].size(); ++neuronIndex) {
						for (int nextNeuronIndex = 0; nextNeuronIndex < w[layerIndex][neuronIndex].size(); ++nextNeuronIndex) {
							//assert(layerIndex < h.size() && neuronIndex < h[layerIndex].size());  // 确保 h 的索引在范围内
							//assert(neuronIndex < w[layerIndex].size() && nextNeuronIndex < w[layerIndex][neuronIndex].size());
							//w[layerIndex][neuronIndex][nextNeuronIndex] += learning * h[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
							w[layerIndex][neuronIndex][nextNeuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
							//printf("w is good!\n");
						}
						//b[layerIndex][neuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
						//printf("finished cal w%d%d\n", layerIndex, neuronIndex);
					}
					//printf("finish cal w%d\n", layerIndex);
					//assert(layerIndex < b.size());  // 确保偏置索引在范围内
					for (int biasIndex = 0; biasIndex < b[layerIndex].size(); ++biasIndex) {
						b[layerIndex][biasIndex] += learning * layerGradients[layerGradients.size() - 1 - layerIndex][biasIndex];
						//assert(layerIndex < layerGradients.size() && layerIndex >= 0);
						//if (biasIndex >= layerGradients[layerIndex].size())
						//  printf("bias index(%d) is larger than the size(%d)!\n", biasIndex, layerGradients[layerIndex].size());
						//assert(biasIndex < layerGradients[layerIndex].size());

						//b[layerIndex][biasIndex] += learning * layerGradients[layerIndex][biasIndex];
					}
					//printf("finish cal b%d\n", layerIndex);

				}
				//printf("finish cal w\n");
			}
			// 每个epoch后输出损失
			//continue;
			if (epoch % LR_VOKE) continue;
			//printf("train-printloss\n");
			double loss = 0;
			for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
				my_vector& output = forward(data[dataIndex]);
				for (int outputIndex = 0; outputIndex < output.size(); ++outputIndex) {
					//double error = label[dataIndex] - output[outputIndex];
					//loss += error * error;  // MSE
					loss += getMSELoss(label[dataIndex][outputIndex], output[outputIndex]);
				}
			}
			loss /= data.size();
			loss /= data[0].size();
			std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
			/*
			for (int i = 0; i < w.size(); ++i) {
			for (int j = 0; j < w[i].size(); ++j) {
			printf("w%d%d:\t", i, j);
			for (int k = 0; k < w[i][j].size(); ++k) {
			printf("%f\t", w[i][j][k]);
			}
			printf("\nb: %f\n", b[i][j]);
			}
			printf("\n");
			}*/

		}

		double loss = 0;
		for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
			my_vector& output = forward(data[dataIndex]);
			for (int outputIndex = 0; outputIndex < output.size(); ++outputIndex) {
				//double error = label[dataIndex] - output[outputIndex];
				//loss += error * error;  // MSE
				loss += getMSELoss(label[dataIndex][outputIndex], output[outputIndex]);
			}
		}
		loss /= data.size();
		loss /= data[0].size();
		std::cout << "Loss: " << loss << std::endl;
	}

	/*
	预测输出。
	@ param
	input：一个一维数组。长度需要与layer第一层（h[0]）的长度相同。倘若不同，程序将会退出。
	@ return
	返回：输出（最后一层layer，即h.back()）的引用（这将导致layer最后一层第一个数值(h[h.size()-1][0]）被改变。
	因而不推荐这个函数。
	*/
	my_vector& predict(my_vector& input)
	{
		// 使用forward函数获取网络的输出
		my_vector& output = forward(input);

		// 这里我们假设网络输出是二分类的概率，使用0.5作为阈值
		// 如果网络设计为多分类，可能需要选择最大值所在的索引
		// 因为我们预期输出是单个值，我们将使用output[0]作为预测概率
		//double threshold = sigmoid(0.5);

		double threshold = 0.5;
		//printf("output0 is:%f\n", output[0]);
		for (auto& n : output) {
			if (n > threshold) n = 1.0;
			else n = 0.0;
		}

		// 返回分类决策，是h.back()的引用。
		return output;
		for (auto& wi : w) {
			for (auto& wj : wi) {
				for (auto wk : wj) {
					printf("%f\t", wk);
				}
				printf(";\t");
			}
			printf("\n");
		}

		return output;

	}
	//my_vector predict(my_vector& input);

	/*
	预测输出。
	@ param
	input：一个一维数组。长度需要与layer第一层（h[0]）的长度相同。倘若不同，程序将会退出。
	therehold: 阈值。超过它，则返回真（1.0），否则返回假（0.0）。为了保持可拓展性，返回暂时使用double类型。
	@ return
	返回：输出（最后一层layer，即h.back()）的第一个元素。通常对于只有一个输出的神经网络来讲，这样是可以用的。
	*/
	double predict(my_vector& input, double therehold)
	{
		//threshold = sigmoid(threshold);
		return (forward(input)[0] > therehold ? 1.0 : 0.0);
	}//todo

	//void printLoss();
	void setLR_VOKE(int LR_VOKE)
	{
		this->LR_VOKE = LR_VOKE;
	}

	my_vector encode(my_vector& input) {
		forward(input);
		return this->o[middle];
	}

	my_vector decode(my_vector& input) {
		int i_max = this->h.size() - 1;
		o[middle].assign(input.begin(), input.end());
		for (int i = middle; i < i_max; i++) {
			this->calculateOutput(o[i], h[i + 1], batch_o[i + 1], w[i], b[i + 1], o[i + 1], i + 1 < i_max);
		}
		my_vector&output = this->o[this->o.size() - 1];
		
		return output;
	}

	my_vector binaryDecode(my_vector& input) {
		int i_max = this->h.size() - 1;
		h[middle].assign(input.begin(), input.end());
		for (int i = 1; i < i_max; i++) {
			this->calculateOutput(o[i], h[i + 1], batch_o[i + 1], w[i], b[i + 1], o[i + 1], i + 1 < i_max);
		}
		my_vector& output = this->o[this->o.size() - 1];
		for (auto& n : output) {
			if (n > 0.5) n = 1.0;
			else n = 0.0;
		}
		return output;
	}
};

class PY_LAE {
private:
	LinearAutoEncoder*model;
	
public:
	~PY_LAE() {
		delete model;
	}

	void selfInit(int epochs, double lr, int inputSize, std::vector<int> v) {
		//std::vector<int> v = { hiddenSize };
		this->model = new LinearAutoEncoder(epochs, lr, inputSize, v);
	}

	std::vector<double> predict(std::vector<double>& input) {
		return this->model->predict(input);
	}

	std::vector<double> encode(std::vector<double>& input) {
		return this->model->encode(input);
	}

	std::vector<double> decode(std::vector<double>& input) {
		return this->model->decode(input);
	}

	std::vector<double> binaryDecode(std::vector<double>& input) {
		return this->model->binaryDecode(input);
	}

	void train(std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& label) {
		this->model->train(input, label);
	}

	void setLR_VOKE(int times) {
		this->model->setLR_VOKE(times);
	}
};
#endif
