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
protected:
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
