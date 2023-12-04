#ifndef MY_K_MEANS
#define MT_K_MEANS 1
#include<bits/stdc++.h>

#include<cassert>
#include <random> //随机数生成
#include <ctime>
//#include <stdlib.h>
//#include <stdio.h>
//#include <cstdlib>
//#include<iostream>

typedef std::vector<double> my_vector;
typedef std::vector<std::vector<double>> coordinate;
/**
* 三维数组。
* 第一维度代表分类
* 第二维度代表的是坐标点的集合。
* 例如：
* orderSamples[0]表示的是，所有被分类为第0类的坐标点的集合。
*/
typedef std::vector<coordinate> orderedSample;
typedef std::vector<int> cluster;

//std::default_random_engine generator(static_cast<unsigned>(time(0)));
//std::uniform_real_distribution<double> distribution(-5.0, 5.0); // 例如在-1.0到1.0之间生成随机数



class KMeans {
private:
	int dimension;
	int k;
	coordinate clusterCenters;//中心点mu。 长度为k。 

	/**
	* 整理后的坐标点。表示：
	*/
	orderedSample orderedSamples;//整理后的坐标点，三维向量。

	cluster labels;//标签。长度为k。 
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

			//todo: 切换为随机数px。
			//px = (rand() % (N + 1) / (float)(N + 1))*10;
			//px = distribution(generator);
		}
	}
	
	void update(coordinate& points) {
		//清空orderedSample
		orderedSample o(this->k);
		this->orderedSamples.swap(o);
		//printf("update start!\n");
		int size = points.size();
		//更新每一个点的标签
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
			//将其更新到orderedSamples中
			//printf("pass2\n");
			this->orderedSamples[label].push_back(points[i]);
		}
		//printf("update each label\n");

		//更新聚类中心
		for (int label = 0; label < k; ++label) {
			my_vector v(this->dimension, 0.0);//存储维度为dimension下，所有坐标的和
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
		std::uniform_real_distribution<double> distribution(-5.0, 5.0);			// 例如在-1.0到1.0之间生成随机数

		for(int i = 0; i < k; ++i){
			//initMu(this->clusterCenters[i], );//生成随机中心点坐标，请加入函数方法。 
			//labels[i] = i;//labels从0开始初始化划分。和每个mu值一一对应。 
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
