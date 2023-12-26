#ifndef _ETHY_NEURON_
#define _ETHY_NEURON_ 1

//#include<bits/stdc++.h>
#include<vector>
#include<iostream>
#include <cassert>
#include <stdexcept> // �����׳��쳣
#include <random> //���������
#include <ctime>

typedef std::vector<double> my_vector;// to make it easier for programming.
typedef std::vector<my_vector> single_power;
typedef std::vector<single_power> my_power;

/*
the power between layer (h, i) and layer (h+1, j) will be stored in:
w[h][i][j]. w is an instance of my_power(3d-vector of double)
*/

/*
* Ϊ���������Ԥ��ֵ����MSE��ʧ��
calculate MSE loss for single output and single prediction.
*/
/*
* Ϊ���������Ԥ��ֵ����MSE��ʧ��
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
		// �������wֵ
		// ��ʼ�������������
		std::default_random_engine generator(static_cast<unsigned>(time(0)));
		std::uniform_real_distribution<double> distribution(-1.0, 1.0); // ������-1.0��1.0֮�����������

		// ��ʼ��Ȩ�ؾ���
		this->w = my_power(this->h.size() - 1);
		for (int i = 0; i < this->w.size(); ++i) {
			this->w[i] = single_power(h[i].size(), my_vector(h[i + 1].size()));
			for (int j = 0; j < w[i].size(); ++j) {
				for (int k = 0; k < w[i][j].size(); ++k) {
					w[i][j][k] = distribution(generator); // ʹ����������
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
		// ȷ��Ȩ�ؾ������������������ĳߴ���ƥ��
		//for (int i = 0; i < power.size(); ++i) {
		//assert(power[i].size() == y.size());
		//}

		// ȷ��ƫ�������ĳߴ�����������ĳߴ���ƥ��
		//assert(bias.size() == y.size());
		//my_vector x = x0;

		orth(x, x);
		// �������������ÿ��Ԫ��
		int y_size = y.size();
		for (int j = 0; j < y_size; ++j) {
			y[j] = 0.0;
			for (int k = 0; k < x.size(); k++) {
				// ȷ��Ȩ�ؾ�������������������ĳߴ���ƥ��
				//assert(k < power.size());
				// ȷ����ǰ��k����һ����Ӧ��j��
				//assert(j < power[k].size());
				y[j] += x[k] * power[k][j];
			}
			//����y
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
	epoches: ��������
	lr: ѧϰ��
	h: ���������롢����������layer��ȡ���ʱ����ȡh.back()[0]��Ϊ����ο���
	@ Deprecated
	*/
	LinearAutoEncoder(int epoches, double lr, std::vector<my_vector> h)
	{
		this->init(epoches, lr, h);
	}
	//MyNeuron(int epoches, double lr, std::vector<std::)

	/*
	@ params:
	epoches: ��������
	lr: ѧϰ��
	inputSize: ����ά��
	hiddenLayerSizes: �������ĵĳ��Ƚ��������ز�ĳ��ȣ���������е�ÿ��Ԫ�ؽ�����ÿ�����ز��ά�ȡ�
		��������{2,3,3}, ���ز��һ����Ԫ������2���ڶ���������Ԫ��������3.

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
	epoches: ��������
	lr: ѧϰ��
	inputSize: ����ά��
	hiddenLayerSizes: �������ĵĳ��Ƚ��������ز�ĳ��ȣ���������е�ÿ��Ԫ�ؽ�����ÿ�����ز��ά�ȡ�
		��������{2,3,3}, ���ز��һ����Ԫ������2���ڶ���������Ԫ��������3.
	hidenSize: ���ڲ���������һ�����飬������Ҫ����һ�����飨���ز��������ĳ��ȡ�
	@ comments
	���������Ҫ��������python�����ڷ�ֹC++��vector���ݺ�python��Ԫ��Ĵ��ݳ��ֵ����⡣
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
	�������
	*/
	virtual double sigmoid(double x)
	{
		//return x;
		//return x > 0 ? x : 0.01*x;
		//return x;
		return 1 / (1 + exp(-x));
	}

	/*
	������ĵ�����
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
		// ȷ���������ݵĳߴ������������ĳߴ���ƥ��
		//if (data.size() != h[0].size()) {
		//throw std::invalid_argument("Size of input data does not match the size of the network's input layer.");
		//}

		// ���������ݳ�ʼ����һ������
		h[0].assign(data.begin(), data.end());
		o[0].assign(data.begin(), data.end());//����㣬����Ҫsigmoid
		// ����ǰ�򴫲�
		int i_max = this->h.size() - 1;
		for (int i = 0; i < i_max; i++) {
			// �ڽ��о���˷�֮ǰ��ȷ��������Ч
			//if (i >= w.size() || i + 1 >= h.size() || i + 1 >= b.size()) {
			//throw std::out_of_range("Index out of range during forward pass.");
			//}

			// ���Ȩ�ؾ����ά���Ƿ���ȷ
			//if (h[i].size() != w[i].size()) {
			//    throw std::invalid_argument("Mismatch between layer output size and weight matrix size.");
			//}

			// ���Ȩ�ؾ����ÿ�������ĳߴ��Ƿ�����һ��ĳߴ�ƥ��
			//for (int k = 0; k < w[i].size(); ++k) {
			//if (w[i][k].size() != h[i + 1].size()) {
			//throw std::invalid_argument("Mismatch between weight matrix size and next layer size.");
			//}
			//}

			// ������һ������
			//this->calculateOutput(h[i], h[i + 1], w[i], b[i + 1]);
			this->calculateOutput(o[i], h[i + 1], batch_o[i + 1], w[i], b[i + 1], o[i + 1], i + 1 < i_max);
			//printf("new h will be:\n");
			//for (int ti = 0; ti < h[i+1].size(); ++ti) {

			//printf("h%d%d is: %f\t", i+1, ti, h[i+1][ti]);
			//}
		}

		// �������һ������
		//return this->h[this->h.size() - 1];
		return this->o[this->o.size() - 1];
	}
	//my_vector forward(std::vector<my_vector>& data);

	/*
	ѵ��������Ȩ��w��ƫ��b��
	�ú��������data��label����д�������
	@ params
	data: ��ά��m*n��ѵ�����顣m��ʾ���������� n��ʾÿ������ά���ġ�
		��Ҫע����ǣ�����ά��n��Ҫ��layer�ĵ�һ�㣨��h[0]����ά����ͬ�����������˳���
	label�����������Ĳο��������һ��һά�ĳ���Ϊm��double���顣
	*/
	void train(std::vector<my_vector>& data, my_vector& label) {
		//assert(data.size() == label.size());  // ȷ�����ݺͱ�ǩ������ƥ��
		for (int epoch = 0; epoch < epoches; ++epoch) {
			for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
				//assert(dataIndex < label.size());  // ȷ����ǩ�����ڷ�Χ��

				//printf("train-forward\n");
				// ǰ�򴫲�
				my_vector output = forward(data[dataIndex]);
				my_vector& output_h = this->h[h.size() - 1];

				//assert(!output.empty());  // ȷ�������Ϊ��
				//printf("train-gradient\n");
				// �����������ݶ�
				my_vector outputLayerGradient;

				for (int neuronIndex = 0; neuronIndex < 1; ++neuronIndex) {
					double error = label[dataIndex] - output[neuronIndex];
					//outputLayerGradient.push_back(error * d_sigmoid(output[neuronIndex]));
					//if (error >= 0) error = 0;
					outputLayerGradient.push_back(error * d_sigmoid(output_h[neuronIndex]));
				}
				//printf("train-backward\n");
				// ���򴫲�
				std::vector<my_vector> layerGradients;
				layerGradients.push_back(outputLayerGradient);
				for (int layerIndex = h.size() - 2; layerIndex >= 0; --layerIndex) {
					//assert(layerIndex < w.size());  // ȷ��Ȩ�������ڷ�Χ��
					my_vector layerGradient;
					for (int neuronIndex = 0; neuronIndex < h[layerIndex].size(); ++neuronIndex) {
						double gradientSum = 0;
						for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < h[layerIndex + 1].size(); ++nextLayerNeuronIndex) {
							//assert(layerIndex < w.size() && neuronIndex < w[layerIndex].size() && nextLayerNeuronIndex < w[layerIndex][neuronIndex].size()); // ȷ��Ȩ�������ڷ�Χ��
							gradientSum += w[layerIndex][neuronIndex][nextLayerNeuronIndex] * layerGradients.back()[nextLayerNeuronIndex];
							//if (gradientSum >= 0) gradientSum = 0;//�ݶȼ���
						}
						layerGradient.push_back(gradientSum * d_sigmoid(h[layerIndex][neuronIndex]));
					}
					layerGradients.push_back(layerGradient);
				}
				//printf("train-re-new\n");
				// ����Ȩ�غ�ƫ��
				for (int layerIndex = 0; layerIndex < w.size(); ++layerIndex) {
					for (int neuronIndex = 0; neuronIndex < w[layerIndex].size(); ++neuronIndex) {
						for (int nextNeuronIndex = 0; nextNeuronIndex < w[layerIndex][neuronIndex].size(); ++nextNeuronIndex) {
							//assert(layerIndex < h.size() && neuronIndex < h[layerIndex].size());  // ȷ�� h �������ڷ�Χ��
							//assert(neuronIndex < w[layerIndex].size() && nextNeuronIndex < w[layerIndex][neuronIndex].size());
							//w[layerIndex][neuronIndex][nextNeuronIndex] += learning * h[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
							w[layerIndex][neuronIndex][nextNeuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
							//printf("w is good!\n");
						}
						//b[layerIndex][neuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
						//printf("finished cal w%d%d\n", layerIndex, neuronIndex);
					}
					//printf("finish cal w%d\n", layerIndex);
					//assert(layerIndex < b.size());  // ȷ��ƫ�������ڷ�Χ��
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
			// ÿ��epoch�������ʧ
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
		//assert(data.size() == label.size());  // ȷ�����ݺͱ�ǩ������ƥ��
		for (int epoch = 0; epoch < epoches; ++epoch) {
			for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
				//assert(dataIndex < label.size());  // ȷ����ǩ�����ڷ�Χ��

				//printf("train-forward\n");
				// ǰ�򴫲�
				my_vector output = forward(data[dataIndex]);
				my_vector& output_h = this->h[h.size() - 1];

				//assert(!output.empty());  // ȷ�������Ϊ��
				//printf("train-gradient\n");
				// �����������ݶ�
				my_vector outputLayerGradient;

				for (int neuronIndex = 0; neuronIndex < output_h.size(); ++neuronIndex) {
					double error = label[dataIndex][neuronIndex] - output[neuronIndex];
					//outputLayerGradient.push_back(error * d_sigmoid(output[neuronIndex]));
					outputLayerGradient.push_back(error * d_sigmoid(output_h[neuronIndex]));
				}
				//printf("train-backward\n");
				// ���򴫲�
				std::vector<my_vector> layerGradients;
				layerGradients.push_back(outputLayerGradient);
				for (int layerIndex = h.size() - 2; layerIndex >= 0; --layerIndex) {
					//assert(layerIndex < w.size());  // ȷ��Ȩ�������ڷ�Χ��
					my_vector layerGradient;
					for (int neuronIndex = 0; neuronIndex < h[layerIndex].size(); ++neuronIndex) {
						double gradientSum = 0;
						for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < h[layerIndex + 1].size(); ++nextLayerNeuronIndex) {
							//assert(layerIndex < w.size() && neuronIndex < w[layerIndex].size() && nextLayerNeuronIndex < w[layerIndex][neuronIndex].size()); // ȷ��Ȩ�������ڷ�Χ��
							gradientSum += w[layerIndex][neuronIndex][nextLayerNeuronIndex] * layerGradients.back()[nextLayerNeuronIndex];
						}
						layerGradient.push_back(gradientSum * d_sigmoid(h[layerIndex][neuronIndex]));
					}
					layerGradients.push_back(layerGradient);
				}
				//printf("train-re-new\n");
				// ����Ȩ�غ�ƫ��
				for (int layerIndex = 0; layerIndex < w.size(); ++layerIndex) {
					for (int neuronIndex = 0; neuronIndex < w[layerIndex].size(); ++neuronIndex) {
						for (int nextNeuronIndex = 0; nextNeuronIndex < w[layerIndex][neuronIndex].size(); ++nextNeuronIndex) {
							//assert(layerIndex < h.size() && neuronIndex < h[layerIndex].size());  // ȷ�� h �������ڷ�Χ��
							//assert(neuronIndex < w[layerIndex].size() && nextNeuronIndex < w[layerIndex][neuronIndex].size());
							//w[layerIndex][neuronIndex][nextNeuronIndex] += learning * h[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
							w[layerIndex][neuronIndex][nextNeuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
							//printf("w is good!\n");
						}
						//b[layerIndex][neuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
						//printf("finished cal w%d%d\n", layerIndex, neuronIndex);
					}
					//printf("finish cal w%d\n", layerIndex);
					//assert(layerIndex < b.size());  // ȷ��ƫ�������ڷ�Χ��
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
			// ÿ��epoch�������ʧ
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
	Ԥ�������
	@ param
	input��һ��һά���顣������Ҫ��layer��һ�㣨h[0]���ĳ�����ͬ��������ͬ�����򽫻��˳���
	@ return
	���أ���������һ��layer����h.back()�������ã��⽫����layer���һ���һ����ֵ(h[h.size()-1][0]�����ı䡣
	������Ƽ����������
	*/
	my_vector& predict(my_vector& input)
	{
		// ʹ��forward������ȡ��������
		my_vector& output = forward(input);

		// �������Ǽ�����������Ƕ�����ĸ��ʣ�ʹ��0.5��Ϊ��ֵ
		// ����������Ϊ����࣬������Ҫѡ�����ֵ���ڵ�����
		// ��Ϊ����Ԥ������ǵ���ֵ�����ǽ�ʹ��output[0]��ΪԤ�����
		//double threshold = sigmoid(0.5);

		double threshold = 0.5;
		//printf("output0 is:%f\n", output[0]);
		for (auto& n : output) {
			if (n > threshold) n = 1.0;
			else n = 0.0;
		}

		// ���ط�����ߣ���h.back()�����á�
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
	Ԥ�������
	@ param
	input��һ��һά���顣������Ҫ��layer��һ�㣨h[0]���ĳ�����ͬ��������ͬ�����򽫻��˳���
	therehold: ��ֵ�����������򷵻��棨1.0�������򷵻ؼ٣�0.0����Ϊ�˱��ֿ���չ�ԣ�������ʱʹ��double���͡�
	@ return
	���أ���������һ��layer����h.back()���ĵ�һ��Ԫ�ء�ͨ������ֻ��һ������������������������ǿ����õġ�
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
