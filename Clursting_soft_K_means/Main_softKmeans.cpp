#include <iostream>
#include "softKmeans.h" // Make sure to include the correct header file for SoftKMeans

int main332211() {
    int k = 3; // Number of clusters
    int dimensions = 2; // Number of dimensions for each point
    double beta = 1.0; // Softness parameter
    int numPoints = 100; // Number of data points
    int epochs = 10; // Number of iterations for training

    KMeans *skmeans = new SoftKMeans(k, dimensions, beta);
    

    // Generate some random data points
    std::vector<std::vector<double>> points;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-10.0, 10.0);

    for (int i = 0; i < numPoints; ++i) {
        std::vector<double> point;
        for (int j = 0; j < dimensions; ++j) {
            point.push_back(distribution(generator));
        }
        points.push_back(point);
    }

    // Train the model
    skmeans->train(points, epochs);

    // Print the cluster centers
    std::cout << "Cluster Centers:" << std::endl;
    skmeans->printCenters();

    // Optionally, you can also print the responsibilities matrix
    std::cout << "Responsibilities:" << std::endl;
    // Assuming you have a method in SoftKMeans to get the responsibilities matrix
    auto responsibilities = ((SoftKMeans*)skmeans)->getResponsibilities();
    for (const auto& row : responsibilities) {
        for (double resp : row) {
            std::cout << resp << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
