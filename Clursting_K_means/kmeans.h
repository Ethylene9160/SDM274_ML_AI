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
protected:
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
	
	double distance(my_vector& x, my_vector& y);

	double loss(coordinate& points);
	
	void initMu(my_vector& point);
	
	void update(coordinate& points);
public:
	KMeans(int k, int dimension);
	
	virtual void train(coordinate& points, int epochs);

	void printCenters();
	
	int predict(my_vector& singlePoint);
	
	//~KMeans() ;
		
};

#endif
