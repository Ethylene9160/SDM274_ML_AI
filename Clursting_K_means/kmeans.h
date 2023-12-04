#ifndef MY_K_MEANS
#define MT_K_MEANS 1
#include<bits/stdc++.h>

#include<cassert>
#include <random> //���������
#include <ctime>
//#include <stdlib.h>
//#include <stdio.h>
//#include <cstdlib>
//#include<iostream>

typedef std::vector<double> my_vector;
typedef std::vector<std::vector<double>> coordinate;
/**
* ��ά���顣
* ��һά�ȴ������
* �ڶ�ά�ȴ�����������ļ��ϡ�
* ���磺
* orderSamples[0]��ʾ���ǣ����б�����Ϊ��0��������ļ��ϡ�
*/
typedef std::vector<coordinate> orderedSample;
typedef std::vector<int> cluster;

//std::default_random_engine generator(static_cast<unsigned>(time(0)));
//std::uniform_real_distribution<double> distribution(-5.0, 5.0); // ������-1.0��1.0֮�����������



class KMeans {
private:
	int dimension;
	int k;
	coordinate clusterCenters;//���ĵ�mu�� ����Ϊk�� 

	/**
	* ����������㡣��ʾ��
	*/
	orderedSample orderedSamples;//����������㣬��ά������

	cluster labels;//��ǩ������Ϊk�� 
	/**
	 * @brief calculate distance between x and y
	 * 
	 * @return (double) distance between 2 point x and y.calculate distanc
	 **/
	double squareDistance(my_vector& x, my_vector& y);
	
	double distance(my_vector&x, my_vector&y){
		assert(x.size()== y.size());
		double output = squareDistance(x, y);
		return sqrt(output);
	}

	double loss(coordinate&points){
		int size = points.size();
		assert(size == clusterCenters.size());
		double out = 0;
		for(int i = 0; i < size; ++i){
			out += squareDistance(points[i], clusterCenters[i]);
		}
		return out;
	} 
	
	void initMu(my_vector&point){
		//int N = 999;
		return;
		//srand(time(NULL));
		for(double&px:point){

			//todo: �л�Ϊ�����px��
			//px = (rand() % (N + 1) / (float)(N + 1))*10;
			//px = distribution(generator);
		}
	}
	
	void update(coordinate& points) {
		//���orderedSample
		orderedSample o(this->k);
		this->orderedSamples.swap(o);
		//printf("update start!\n");
		int size = points.size();
		//����ÿһ����ı�ǩ
		for (int i = 0; i < size; ++i) {
			int label = 0;
			double d2 = squareDistance(points[i], this->clusterCenters[0]), tmp = d2;
			for (int j = 1; j < k; ++j) {
				tmp = squareDistance(points[i], this->clusterCenters[j]);
				if (tmp < d2) {
					d2 = tmp;
					label = j;
				}
			}
			this->labels[i] = label;
			//������µ�orderedSamples��
			//printf("pass2\n");
			this->orderedSamples[label].push_back(points[i]);
		}
		//printf("update each label\n");

		//���¾�������
		for (int label = 0; label < k; ++label) {
			my_vector v(this->dimension, 0.0);//�洢ά��Ϊdimension�£���������ĺ�
			for (my_vector&point:this->orderedSamples[label]) {
				for (int i = 0; i < dimension; ++i) {
					v[i] += point[i];
				}
			}
			int size = this->orderedSamples[label].size();
			if (size == 0) continue;
			for (size_t i = 0; i < this->dimension; i++)
			{
				//if (size > 0)
				this->clusterCenters[label][i] = v[i] / size;
				//else this->clusterCenters[label][i] = 0;
			}
		}
	}
public:
	KMeans(int k, int dimension){
		this->k = k;
		this->dimension = dimension;
		//this->labels = std::vector<int>(k);
		this->clusterCenters = std::vector<std::vector<double>>(k, std::vector<double>(dimension));
		this->orderedSamples = orderedSample(k);
		printf("pass\n");

		std::default_random_engine generator(static_cast<unsigned>(time(0)));	//
		std::uniform_real_distribution<double> distribution(-5.0, 5.0);			// ������-1.0��1.0֮�����������

		for(int i = 0; i < k; ++i){
			//initMu(this->clusterCenters[i], );//����������ĵ����꣬����뺯�������� 
			//labels[i] = i;//labels��0��ʼ��ʼ�����֡���ÿ��muֵһһ��Ӧ�� 
			for (auto& px : clusterCenters[i]) {
				px = distribution(generator);
			}
		}
		printf("clusterCenters: \n");
		for (auto& a1 : this->clusterCenters) {
			for (auto& a2 : a1) {
				printf("%f, ", a2);
			}
			printf("\n");
		}	
	}
	
	void train(coordinate&points, int epochs){
		this->labels = std::vector<int>(points.size());
		for(int i = 0; i < epochs; ++i){
			update(points);
		}
	}

	void printCenters() {
		for (auto&center : clusterCenters) {
			printf("[");
			for (auto& n : center) {
				printf("%f, ", n);
			}
			printf("]\n");
		}
	}
	
	int predict(my_vector& singlePoint) {
		std::vector<double> lists = std::vector<double>(k);
		double out = squareDistance(singlePoint, clusterCenters[0]);
		int index = 0;
		for (int i = 1; i < k; ++i) {
			double tmp = squareDistance(singlePoint, clusterCenters[i]);
			if (tmp < out)
			{
				out = tmp;
				index = i;
			}
		}
		return index;
	}
	
	//~KMeans() ;
		
};

#endif
