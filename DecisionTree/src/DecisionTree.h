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
	/*训练数据集
			  trainData
	feature1 |feature2 |feature3
	value01  |value02  |value03
	value11  |value12  |value13
	...
	*/
	std::vector<std::vector<int>> trainData;
	/*每条训练数据一一对应的标签
	[labelA, labelB, labelC,...]
	*/
	std::vector<int> trainLabel;
	/*
	<key, value>=<特征, 特征对应的所有特征值集合>
	例如，trainData中：
			trainData
	feature1(0)	|featur2(1)	|feature3(2)
	0			|2			|0
	1			|0			|1
	1			|0			|1
	2			|1			|1
	那么，键值对将会是：
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
	后文中，大量的变量dataIndexs，即为上文中的Index。用来标记trainData和trainLabel的下标。
	*/

	Tree root;

	void destroyTree(Tree root);

	void printTree(Node* node);

	void loadData(const std::vector<std::vector<int>>& trainData, const std::vector<int>& trainLabel);

	/*
	计算传入数据的标签情况。标签i在数据中出现了几次，就会在key值为i的map中保存出现的次数。
	例如，传入的数据：
	下标		标签
	 0		 0
	 1		 0
	 2		 2
	 3		 1
	 4		 1
	 6		 1
	 9		 1

	返回的map键值对将会是：
	<0,	3>
	<1,	3>
	<2,	1>
	@param dataIndexs: 需要用于计算的trainData或trainLabel的下标
	*/
	std::map<int, int>* featureValueCount(std::vector<int>& dataIndexs);	
	/*
	计算特征为feature下的信息增益。
	*/
	double calculateGain(std::vector<int>& dataIndexs, int feature);

	/*
	计算选中数据的熵。
	@param dataIndexs: 待用于计算的训练数据的下标
	*/
	double entropy(std::vector<int>& dataIndexs);

	/*
	* 提取训练数据中，下标在dataIndexs内的feature特征下特征值为value的数据
	* 
	@param dataIndexs:			用于计算的训练数据的下标
	@param feature：				特征
	@param value：				特征为feature下的特定特征值value
	@return std::vector<int>:	返回传入训练下标dataIndex中，feature特征下特征值为value的所有下标。
	*/
	std::vector<int> splitDataset(std::vector<int>& dataIndexs, int feature, int value);

	/*
	* 返回传入的labelCount中，label出现次数最多的label。
	* 例如：
	* <k,	v>
	* <0,	2>
	* <1,	4>
	* <2,	1>
	* 将会返回1，因为1出现了4次。如果有相同的v，会返回后一个的featureValue。
	* 调用这个函数时，通常使用的map键值对为：key为特征值，value为这个特征值出现的频数。
	* 例如例子中，晴天出现了5次，雨天4次，多云5次。
	* <晴天,	5>
	* <雨天,	4>
	* <多云,	5>
	* 会返回多云。
	*/
	int getMaxFeatureValue(std::map<int, int>& featureValueCount);

	/*
	* @param gains: 
	* <key,value>=<特征, 下一个节点使用该特征的信息增益>
	* 返回value最大时候对应的key，即返回最大信息增益时所使用的feature。
	*/
	int getMaxGainFeature(std::map<int, double>& gains);

	Tree creatTree(std::vector<int>& dataIndexs, std::vector<int>& features);

	int classify(std::vector<int>& testData, Tree root);

public:
	DecisionTree(std::vector<std::vector<int>>& trainData, std::vector<int>& trainLabel);

	~DecisionTree();

	/*
	* 传入一条数据，返回它的预测类别。
	*/
	int classify(std::vector<int>& testData);

	Node* getRoot();

	void printTree();
};
#endif
