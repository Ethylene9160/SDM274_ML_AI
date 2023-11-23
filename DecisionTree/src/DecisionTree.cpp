#include "DecisionTree.h"


void DecisionTree::destroyTree(Tree root)
{
	for (Node* node : root->branches) {
		destroyTree(node);
	}
	delete root;
}

void DecisionTree::printTree(Node* node)
{
	printf("node: attr %d attrval %d\n", node->feature, node->feature_value);
	if (node->isLeaf) {
		printf("leaf: %d\n", node->result);
		return;
	}
	for (Node* cl : node->branches) {
		printTree(cl);
	}
	printf("back\n");
}

void DecisionTree::loadData(const std::vector<std::vector<int>>& trainData, const std::vector<int>& trainLabel)
{
	assert(trainData.size() == trainLabel.size());//保证数组最外层维度相同

	this->trainData = trainData;
	this->trainLabel = trainLabel;

	for (auto& data : this->trainData) {
		//取出trainData的某一个。例如[1,0,2,3]
		int size = data.size();
		for (int i = 0; i < size; ++i) {
			featureValues[i].insert(data[i]);
		}
		/*
		最终将会构成：
		feature0：0 1 2 3
		feature1: 1 2  3
		feature2: 0 1 2 3
		*/
	}
}

DecisionTree::DecisionTree(std::vector<std::vector<int>>& trainData, std::vector<int>& trainLabel)
{
	loadData(trainData, trainLabel);
	std::vector<int> dataIndexs(trainData.size());//数据集
	for (int i = 0; i < trainData.size(); i++) {
		dataIndexs[i] = i;
	}
	//更正：不是所有的feature都是严格的从0开始，逐一递增。
	std::set<int> fvs;
	for (auto& tds : trainData) {
		for (int ft : tds) {
			fvs.insert(ft);
		}
	}
	//std::vector<int> features(fvs.begin(), fvs.end());//属性集合
	std::vector<int> features(trainData[0].size());
	for (int i = 0; i < trainData[0].size(); i++) {
		features[i] = i;
	}
	printf("dataIndexs:\n");
	for (auto& data : dataIndexs) {
		printf("%d\t", data);
	}
	printf("\nfeautres:\n");
	for (auto& ft : features) {
		printf("%d\t", ft);
	}
	printf("\n");
	this->root = creatTree(dataIndexs, features);//创建决策树
	
}

DecisionTree::~DecisionTree()
{
	destroyTree(root);
}

std::map<int, int>* DecisionTree::featureValueCount(std::vector<int>& dataIndexs)
{
	//std::unique_ptr<std::map<int, int>> maxFeatureValue;
	std::map<int, int>*res = new std::map<int,int>();
	for (int index : dataIndexs) {
		(*res)[trainLabel[index]]++;
	}
	return res;
}

double DecisionTree::entropy(std::vector<int>& dataIndexs)
{
	std::map<int, int>*label_count = featureValueCount(dataIndexs);
	int len = dataIndexs.size();
	double result = 0.0;
	for (auto& count : *label_count) {
		if (count.second == 0.0) 
		{
			//result -= 0;
			continue;
		}
		double pi = count.second / static_cast<double>(len);
		result -= pi * log2(pi);
	}
	delete label_count;
	return result;
}

std::vector<int> DecisionTree::splitDataset(std::vector<int>& dataIndexs, int feature, int value)
{
	std::vector<int> res;
	for (int index : dataIndexs) {
		//如果数据中存在feature特征的
		if (trainData[index][feature] == value) {
			res.push_back(index);
		}
	}
	return res;
}

int DecisionTree::getMaxFeatureValue(std::map<int, int>& featureValueCount)
{
	int max_count = 0;
	int maxFeatureValue = 0;
	for (auto& label : featureValueCount) {
		if (max_count <= label.second) {
			max_count = label.second;
			maxFeatureValue = label.first;
		}
	}
	return maxFeatureValue;
}

double DecisionTree::calculateGain(std::vector<int>& dataIndexs, int feature)
{
	std::set<int>& values = featureValues[feature];//获取这个特征下，有哪些特征值。例如，特征值为“天气”，那么获得到的values则代表“晴天”“阴天”“雨天”。
	double result = 0.0;
	for (int val : values) {
		std::vector<int> subDataIndexs = splitDataset(dataIndexs, feature, val);//获取dataIndex中，特征为feature，特征值为val的数据下标。
		result += subDataIndexs.size() / double(dataIndexs.size()) * entropy(subDataIndexs);
	}
	return entropy(dataIndexs) - result;
}

int DecisionTree::getMaxGainFeature(std::map<int, double>& gains)
{
	double max_gain = 0;
	int max_gain_feature = 0;
	for (auto& gain : gains) {
		if (max_gain <= gain.second) {
			max_gain = gain.second;
			max_gain_feature = gain.first;
		}
	}
	return max_gain_feature;
}

Tree DecisionTree::creatTree(std::vector<int>& dataIndexs, std::vector<int>& features)
{
	Tree root = new Node();
	std::map<int, int>* featureValue_count = featureValueCount(dataIndexs);
	//由于每次往下建立一个节点，都意味着特征被使用了一次。往后建立的过程中，特征数量将会逐层减少。
	//如果还需要进行分类构造节点的特征大小为0，也就是说在建立过程中，已经完成了所有的特征变量的建立，这意味着这将会是个叶子节点：
	if (features.size() == 0) {
		root->result = getMaxFeatureValue(*featureValue_count);//结果将会是dataIndexs中出现频数最高的那个特征对应的label。
		root->isLeaf = 1;
		delete featureValue_count;
		return root;
	}

	//如果这一个dataIndexs的特征值都相同，即featureValue_count这一个map<int, int>中只有一个键值对，说明这个特征条件下，只剩了这一种。
	if (featureValue_count->size() == 1) {
		root->result = featureValue_count->begin()->first;
		root->isLeaf = 1;
		delete featureValue_count;
		return root;
	}

	//计算按照feature中不同情况下分类时候的信息增益
	std::map<int, double> gains;
	for (int ft : features) {
		gains[ft] = calculateGain(dataIndexs, ft);
	}

	//获取信息增益最大的特征类别
	int max_gain_feature = getMaxGainFeature(gains);

	//接下来，我们需要将这个信息增益最大的特征类别作为传输节点，然后再次使用这样的方法，对这个节点进行再分类。
	std::vector<int> subFeatures = features;
	subFeatures.erase(find(subFeatures.begin(), subFeatures.end(), max_gain_feature));//移除subFeatures下的max_gain_feature

	//对信息增益最大的特征类别的特征值，分别建立节点，然后继续下推。
	for (int val : featureValues[max_gain_feature]) {
		Tree branch = new Node();
		//将dataIndexs中，特征为max_gain_feature, 值为val的data挑选出来，变成新的数据集。
		std::vector<int> subDataset = splitDataset(dataIndexs, max_gain_feature, val);

		//如果训练集中没有满足要求的数据，说明训练集中没有符合条件的数据，那么对其进行特殊处理，将其单独作为一个叶子节点返回。
		//节点的值就是我们所使用的特征中，出现次数最多的特征值。
		//通常会由于数据集不够完善，会出现这个情况。
		//在这门课程中给出的测试样例lenses.data中，没有这种情况。
		if (subDataset.size() == 0) {
			branch->isLeaf = true;
			branch->result = getMaxFeatureValue(*featureValue_count);
			branch->feature = max_gain_feature;
			branch->feature_value = val;
			root->branches.push_back(branch);
		}
		//递归建树
		else {
			branch = creatTree(subDataset, subFeatures);
			branch->feature = max_gain_feature;
			branch->feature_value = val;
			root->branches.push_back(branch);
		}
	}
	delete featureValue_count;
	return root;
}

int DecisionTree::classify(std::vector<int>& testData, Tree root)
{
	//printf("attr: %d, attrval: %d\n", root->feature, root->feature_value);
	if (root->isLeaf) {//叶子节点，返回其value。
		// printf("feature: %d, attrval: %d, result: %d\n", root->feature, root->feature_value, root->result);
		// printf("result: %d\n", root->result);
		return root->result;
	}
	// printf("\n");
	for (Node* node : root->branches) {
		//找到分支，并在分支中再细分
		//如果训练集中的某个feature（node->feature）的值为这个节点的feature_value，那么就继续往下匹配。
		if (testData[node->feature] == node->feature_value) {
			return classify(testData, node);
		}
	}
	return 0;
}


/*
* 传入一条数据，返回它的预测类别。
*/

int DecisionTree::classify(std::vector<int>& testData)
{
	return classify(testData, this->root);
}

Node* DecisionTree::getRoot()
{
	return root;
}

void DecisionTree::printTree()
{
	printTree(this->root);
}
