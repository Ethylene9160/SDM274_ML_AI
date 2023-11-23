#ifndef _MY_DETERMINE_TREE_
#define _MY_DETERMINE_TREE_
#include <vector>
#include <map>
#include <set>
#include <math.h>
#include <algorithm>
#include <memory>
#include <cassert>

struct Node{
	int result;
	int feature;
	int feature_value;
	std::vector<Node*> branches;
	bool isLeaf;
	Node() :result(-1), feature(-1), feature_value(-1), isLeaf(0) {}
	Node(bool isLeaf, int feature, int feature_value, int res) :result(res), feature(feature), feature_value(feature_value), isLeaf(isLeaf) {}
}; 
typedef Node* Tree;

class DecisionTree {
private:
	/*ѵ�����ݼ�
			  trainData
	feature1 |feature2 |feature3
	value01  |value02  |value03
	value11  |value12  |value13
	...
	*/
	std::vector<std::vector<int>> trainData;
	/*ÿ��ѵ������һһ��Ӧ�ı�ǩ
	[labelA, labelB, labelC,...]
	*/
	std::vector<int> trainLabel;
	/*
	<key, value>=<����, ������Ӧ����������ֵ����>
	���磬trainData�У�
			trainData
	feature1(0)	|featur2(1)	|feature3(2)
	0			|2			|0
	1			|0			|1
	1			|0			|1
	2			|1			|1
	��ô����ֵ�Խ����ǣ�
	<0,{0,1,2}>
	<1,{0,1,2}>
	<2,{0,1>
	*/
	std::map<int, std::set<int>> featureValues;
	/*
	index	|		   trainData			|trainLabel
	  		|feature1 |feature2 |feature3	|  
	  0		|value01  |value02  |value03	|  labelA
	  1		|value11  |value12  |value13	|  labelB
	  ...
	�����У������ı���dataIndexs����Ϊ�����е�Index���������trainData��trainLabel���±ꡣ
	*/

	Tree root;

	void destroyTree(Tree root);

	void printTree(Node* node);

	void loadData(const std::vector<std::vector<int>>& trainData, const std::vector<int>& trainLabel);

	/*
	���㴫�����ݵı�ǩ�������ǩi�������г����˼��Σ��ͻ���keyֵΪi��map�б�����ֵĴ�����
	���磬��������ݣ�
	�±�		��ǩ
	 0		 0
	 1		 0
	 2		 2
	 3		 1
	 4		 1
	 6		 1
	 9		 1

	���ص�map��ֵ�Խ����ǣ�
	<0,	3>
	<1,	3>
	<2,	1>
	@param dataIndexs: ��Ҫ���ڼ����trainData��trainLabel���±�
	*/
	std::map<int, int>* featureValueCount(std::vector<int>& dataIndexs);	
	/*
	��������Ϊfeature�µ���Ϣ���档
	*/
	double calculateGain(std::vector<int>& dataIndexs, int feature);

	/*
	����ѡ�����ݵ��ء�
	@param dataIndexs: �����ڼ����ѵ�����ݵ��±�
	*/
	double entropy(std::vector<int>& dataIndexs);

	/*
	* ��ȡѵ�������У��±���dataIndexs�ڵ�feature����������ֵΪvalue������
	* 
	@param dataIndexs:			���ڼ����ѵ�����ݵ��±�
	@param feature��				����
	@param value��				����Ϊfeature�µ��ض�����ֵvalue
	@return std::vector<int>:	���ش���ѵ���±�dataIndex�У�feature����������ֵΪvalue�������±ꡣ
	*/
	std::vector<int> splitDataset(std::vector<int>& dataIndexs, int feature, int value);

	/*
	* ���ش����labelCount�У�label���ִ�������label��
	* ���磺
	* <k,	v>
	* <0,	2>
	* <1,	4>
	* <2,	1>
	* ���᷵��1����Ϊ1������4�Ρ��������ͬ��v���᷵�غ�һ����featureValue��
	* �����������ʱ��ͨ��ʹ�õ�map��ֵ��Ϊ��keyΪ����ֵ��valueΪ�������ֵ���ֵ�Ƶ����
	* ���������У����������5�Σ�����4�Σ�����5�Ρ�
	* <����,	5>
	* <����,	4>
	* <����,	5>
	* �᷵�ض��ơ�
	*/
	int getMaxFeatureValue(std::map<int, int>& featureValueCount);

	/*
	* @param gains: 
	* <key,value>=<����, ��һ���ڵ�ʹ�ø���������Ϣ����>
	* ����value���ʱ���Ӧ��key�������������Ϣ����ʱ��ʹ�õ�feature��
	*/
	int getMaxGainFeature(std::map<int, double>& gains);

	Tree creatTree(std::vector<int>& dataIndexs, std::vector<int>& features);

	int classify(std::vector<int>& testData, Tree root);

public:
	DecisionTree(std::vector<std::vector<int>>& trainData, std::vector<int>& trainLabel);

	~DecisionTree();

	/*
	* ����һ�����ݣ���������Ԥ�����
	*/
	int classify(std::vector<int>& testData);

	Node* getRoot();

	void printTree();
};
#endif
