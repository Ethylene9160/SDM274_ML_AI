#include<bits/stdc++.h>
#include"LinearAutoEncoder.h"
#include "MyNeuron.h"
#include<stdio.h>
#include<stdlib.h>
using namespace std;
int main() {
	vector<double> input = {
		1,1,1,0,1,1,1,0,1,1,1,0
	};
	vector<vector<double>> inputData = {
		{1,0,1,1,1,1,0,0,1},
		{1,1,0,0,1,1,1,0,1},
		{1,0,1,1,0,0,1,1,0},
		{1,0,1,1,0,0,1,0,0},
		{1,0,1,1,0,0,0,1,1}
	};
	//LinearAutoEncoder encoder(input.size(), 4, 0.01);
	// 
	inputData = vector<vector<double>>(1, input);
	//LinearAutoEncoder encoder(5000, 0.01, inputData[0].size(), {4});
	PY_LAE encoder;
	encoder.selfInit(5000, 0.01, inputData[0].size(), { 3 });
	
	// vector<double> label = { 1,1,1,1,1,1,1,1,1,1 };
	cout << "begin to train" << endl;
	encoder.train(inputData, inputData);
	//cout << encoder.predict(input)[0];
	//return 0;
	printf("predict derectly:\n");
	vector<double> output = encoder.predict(inputData[0]);
	for (auto n : output) {
		printf("%f\t", n);
	}
	printf("\nencode:\n");
	vector<double> decoding = encoder.encode(inputData[0]);
	for (auto n : decoding) {
		printf("%f\t", n);
	}

	printf("\ndecode:\n");
	vector<double> decod = encoder.decode(decoding);
	for (auto n : decod) {
		printf("%f\t", n);
	}
	int corNum=0;
	for (int i = 0; i < inputData[0].size(); ++i) {
		if(inputData[0][i] == decod[i]) corNum += 1;
	}
	printf("\nbcor rate: %f\n", (double)corNum / (double)inputData[0].size());
	return 0;
}

