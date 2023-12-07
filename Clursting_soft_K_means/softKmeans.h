#ifndef ETHY_SOFT_K_MEANS
#define ETHY_SOFT_K_MEANS 1
#include<bits/stdc++.h>

#include "../Clursting_K_means/kmeans.h"

class SoftKMeans : public KMeans {
private:
    std::vector<std::vector<double>> responsibilities;
    double beta; // Softness parameter

    void updateResponsibilities(coordinate& points) {
        this->updateResponsibilities(points, this->responsibilities);
    }

    void updateResponsibilities(coordinate& points, std::vector<std::vector<double>>& responsibilities) {
        for (int i = 0; i < points.size(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < k; ++j) {
                responsibilities[i][j] = exp(-beta * squareDistance(points[i], clusterCenters[j]));
                sum += responsibilities[i][j];
            }
            for (int j = 0; j < k; ++j) {
                responsibilities[i][j] /= sum; // Normalize to get probabilities
            }
        }
    }

    void updateClusterCenters(coordinate& points) {
        for (int j = 0; j < k; ++j) {
            my_vector newCenter(dimension, 0.0);
            double totalResponsibility = 0.0;

            for (int i = 0; i < points.size(); ++i) {
                for (int d = 0; d < dimension; ++d) {
                    newCenter[d] += responsibilities[i][j] * points[i][d];
                }
                totalResponsibility += responsibilities[i][j];
            }

            if (totalResponsibility > 0) {
                for (int d = 0; d < dimension; ++d) {
                    this->clusterCenters[j][d] = newCenter[d] / totalResponsibility;
                }
            }
        }
    }

public:
    SoftKMeans(int k, int dimension, double beta): KMeans(k, dimension), beta(beta) {
       
    }

    /**
    * K有多大，responsibilities矩阵就有多大。
    */
    void train(coordinate& points, int epochs){
        responsibilities = std::vector<std::vector<double>>(points.size(), std::vector<double>(k, 0.0));
        for (int i = 0; i < epochs; ++i) {
            updateResponsibilities(points);
            updateClusterCenters(points);
            // Optionally implement a convergence check here to break early if the centers have stabilized
        }
    }

    std::vector<std::vector<double>>& getResponsibilities() {
        return this->responsibilities;
    }

    std::vector <std::vector<double>> getResponsibilities(coordinate&points) {
        std::vector<std::vector<double>> predictResponsibilities(points.size(), std::vector<double>(k, 0.0));
        updateResponsibilities(points, predictResponsibilities);
        return predictResponsibilities;
    }
};

#endif
