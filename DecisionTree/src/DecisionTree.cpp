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
	assert(trainData.size() == trainLabel.size());//��֤���������ά����ͬ

	this->trainData = trainData;
	this->trainLabel = trainLabel;

	for (auto& data : this->trainData) {
		//ȡ��trainData��ĳһ��������[1,0,2,3]
		int size = data.size();
		for (int i = 0; i < size; ++i) {
			featureValues[i].insert(data[i]);
		}
		/*
		���ս��ṹ�ɣ�
		feature0��0 1 2 3
		feature1: 1 2  3
		feature2: 0 1 2 3
		*/
	}
}

DecisionTree::DecisionTree(std::vector<std::vector<int>>& trainData, std::vector<int>& trainLabel)
{
	loadData(trainData, trainLabel);
	std::vector<int> dataIndexs(trainData.size());//���ݼ�
	for (int i = 0; i < trainData.size(); i++) {
		dataIndexs[i] = i;
	}
	//�������������е�feature�����ϸ�Ĵ�0��ʼ����һ������
	std::set<int> fvs;
	for (auto& tds : trainData) {
		for (int ft : tds) {
			fvs.insert(ft);
		}
	}
	//std::vector<int> features(fvs.begin(), fvs.end());//���Լ���
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
	this->root = creatTree(dataIndexs, features);//����������
	
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
		//��������д���feature������
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
	std::set<int>& values = featureValues[feature];//��ȡ��������£�����Щ����ֵ�����磬����ֵΪ������������ô��õ���values��������족�����족�����족��
	double result = 0.0;
	for (int val : values) {
		std::vector<int> subDataIndexs = splitDataset(dataIndexs, feature, val);//��ȡdataIndex�У�����Ϊfeature������ֵΪval�������±ꡣ
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
	//����ÿ�����½���һ���ڵ㣬����ζ��������ʹ����һ�Ρ��������Ĺ����У������������������١�
	//�������Ҫ���з��๹��ڵ��������СΪ0��Ҳ����˵�ڽ��������У��Ѿ���������е����������Ľ���������ζ���⽫���Ǹ�Ҷ�ӽڵ㣺
	if (features.size() == 0) {
		root->result = getMaxFeatureValue(*featureValue_count);//���������dataIndexs�г���Ƶ����ߵ��Ǹ�������Ӧ��label��
		root->isLeaf = 1;
		delete featureValue_count;
		return root;
	}

	//�����һ��dataIndexs������ֵ����ͬ����featureValue_count��һ��map<int, int>��ֻ��һ����ֵ�ԣ�˵��������������£�ֻʣ����һ�֡�
	if (featureValue_count->size() == 1) {
		root->result = featureValue_count->begin()->first;
		root->isLeaf = 1;
		delete featureValue_count;
		return root;
	}

	//���㰴��feature�в�ͬ����·���ʱ�����Ϣ����
	std::map<int, double> gains;
	for (int ft : features) {
		gains[ft] = calculateGain(dataIndexs, ft);
	}

	//��ȡ��Ϣ���������������
	int max_gain_feature = getMaxGainFeature(gains);

	//��������������Ҫ�������Ϣ�����������������Ϊ����ڵ㣬Ȼ���ٴ�ʹ�������ķ�����������ڵ�����ٷ��ࡣ
	std::vector<int> subFeatures = features;
	subFeatures.erase(find(subFeatures.begin(), subFeatures.end(), max_gain_feature));//�Ƴ�subFeatures�µ�max_gain_feature

	//����Ϣ��������������������ֵ���ֱ����ڵ㣬Ȼ��������ơ�
	for (int val : featureValues[max_gain_feature]) {
		Tree branch = new Node();
		//��dataIndexs�У�����Ϊmax_gain_feature, ֵΪval��data��ѡ����������µ����ݼ���
		std::vector<int> subDataset = splitDataset(dataIndexs, max_gain_feature, val);

		//���ѵ������û������Ҫ������ݣ�˵��ѵ������û�з������������ݣ���ô����������⴦�����䵥����Ϊһ��Ҷ�ӽڵ㷵�ء�
		//�ڵ��ֵ����������ʹ�õ������У����ִ�����������ֵ��
		//ͨ�����������ݼ��������ƣ��������������
		//�����ſγ��и����Ĳ�������lenses.data�У�û�����������
		if (subDataset.size() == 0) {
			branch->isLeaf = true;
			branch->result = getMaxFeatureValue(*featureValue_count);
			branch->feature = max_gain_feature;
			branch->feature_value = val;
			root->branches.push_back(branch);
		}
		//�ݹ齨��
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
	if (root->isLeaf) {//Ҷ�ӽڵ㣬������value��
		// printf("feature: %d, attrval: %d, result: %d\n", root->feature, root->feature_value, root->result);
		// printf("result: %d\n", root->result);
		return root->result;
	}
	// printf("\n");
	for (Node* node : root->branches) {
		//�ҵ���֧�����ڷ�֧����ϸ��
		//���ѵ�����е�ĳ��feature��node->feature����ֵΪ����ڵ��feature_value����ô�ͼ�������ƥ�䡣
		if (testData[node->feature] == node->feature_value) {
			return classify(testData, node);
		}
	}
	return 0;
}


/*
* ����һ�����ݣ���������Ԥ�����
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
