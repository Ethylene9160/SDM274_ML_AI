#include<bits/stdc++.h>
#include "DecisionTree.h"
using namespace std;
int main() {
    //训练数据: 打球与否。
    //四个特征分别为：
    //天气，温度，湿度，有无风。
    vector<vector<int>> trainData = 
    /*
    {
        {1,  1,  1,  1},
        {1,  1,  1,  2},
        {1,  1,  2,  1},
        {1,  1,  2,  2},
        {1,  2,  1,  1},
        {1,  2,  1,  2},
        {1,  2,  2,  1},
        {1,  2,  2,  2},
        {2,  1,  1,  1},
        {2,  1,  1,  2},
        {2,  1,  2,  1},
        {2,  1,  2,  2},
        {2,  2,  1,  1},
        {2,  2,  1,  2},
        {2,  2,  2,  1},
        {2,  2,  2,  2},
        {3,  1,  1,  1},
        {3,  1,  1,  2},
        {3,  1,  2,  1},
        {3,  1,  2,  2},
        {3,  2,  1,  1},
        {3,  2,  1,  2},
        {3,  2,  2,  1},
        {3,  2,  2,  2}
    };

    trainData = 
    */
    {
        {0,0,0,0},
        {0,0,0,1},
        {2,0,0,0},
        {1,2,0,0},
        {1,1,1,0},
        {1,1,1,1},
        {2,1,1,1},
        {0,2,0,0},
        {0,1,1,0},
        {1,2,1,0},
        {0,2,1,1},
        {2,2,0,1},
        {2,0,1,0},
        {1,2,0,1}
    };
    //训练标签
    vector<int> trainLabel = 
    /*{
        3
        ,  2
        ,  3
        ,  1
        ,  3
        ,  2
        ,  3
        ,  1
        ,  3
        ,  2
        ,  3
        ,  1
        ,  3
        ,  2
        ,  3
        ,  3
        ,  3
        ,  3
        ,  3
        ,  1
        ,  3
        ,  2
        ,  3
        ,  3
    };
    trainLabel = */
    {
        0,0,1,1,1,0,1,0,1,1,1,1,1,0
    };


    DecisionTree decisionTree = DecisionTree(trainData, trainLabel);

    //测试
    
    int size = trainData.size();

    decisionTree.classify(trainData[1]);


    int corNum = 0;
    for (int i = 0; i < size; ++i) {
        int bq = decisionTree.classify(trainData[i]);
        if (bq == trainLabel[i]) {
            corNum++;
        }
        printf("index i = %d, value = %d, realLabel is %d\n", i, bq, trainLabel[i]);
    }
    printf("correct rate: %f\n", double(corNum) / double(size));
    
    return 0;
}