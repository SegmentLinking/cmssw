#ifndef DBSCAN_H
#define DBSCAN_H

#pragma once

#include <queue>
#include <cmath>

class DBSCAN {
public:
    DBSCAN(float eps, int minPts);
    int* fit(const float* data, int dataSize, int dataDim);

private:
    float epsilon; // max distance between points to be considered in the same cluster
    int minPoints; // min number of points to form a cluster
    int* labels;

    float euclideanDistance(const float* point1, const float* point2, int dataDim);
    int* regionQuery(const float* data, int pointIdx, int dataSize, int dataDim);
    void expandCluster(const float* data, int pointIdx, int clusterId, int dataSize, int dataDim);
};

#endif