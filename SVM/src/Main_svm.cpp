#include<bits/stdc++.h>
#include"svm.h"
using namespace std;
int main() {
	svm my_svm;
	vector<vector<double>> X = {
		{5.0,1.0},
		{6.0,2.0},
		{5.0,2.0},
		{6.0,1.0},

		{3.0,0.0},
		{2.0,0.0},
		{2.0,-1.0},
		{3.0,-1.0}
	};
	vector<int> y = {
		1,1,1,1,	-1,-1,-1,-1
	};
	my_svm.fit(X, y);
	vector<int> result = my_svm.predict(X);
	for (int& n : result) {
		printf("%d\t", n);
	}
	return 0;
}