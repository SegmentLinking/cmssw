#include "dbscan.h"

#include <iostream>
#include <random>

DBSCAN::DBSCAN(float eps, int minPts) : epsilon(eps), minPoints(minPts) {}

float DBSCAN::euclideanDistance(const float* point1, const float* point2, int dataDim) {
    float sum = 0.0;
    for (int i = 0; i < dataDim; ++i) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return std::sqrt(sum);
}

int* DBSCAN::regionQuery(const float* data, int pointIdx, int dataSize, int dataDim) {
    int* neighbors = new int[dataSize];
    int neighborCount = 0;
    for (int i = 0; i < dataSize; ++i) {
        if (euclideanDistance(&data[pointIdx * dataDim], &data[i * dataDim], dataDim) <= epsilon) {
            neighbors[neighborCount++] = i;
        }
    }
    return neighbors;
}

void DBSCAN::expandCluster(const float* data, int pointIdx, int clusterId, int dataSize, int dataDim) {
    int* seeds = regionQuery(data, pointIdx, dataSize, dataDim);
    int seedCount = dataSize;
    labels[pointIdx] = clusterId;
    std::queue<int> pointsQueue;
    for (int i = 0; i < seedCount; ++i) {
        if (seeds[i] != pointIdx) {
            pointsQueue.push(seeds[i]);
        }
    }
    delete[] seeds;
    
    while (!pointsQueue.empty()) {
        int currentPointIdx = pointsQueue.front();
        pointsQueue.pop();

        if (labels[currentPointIdx] == -1) {
            labels[currentPointIdx] = clusterId;
        }
        if (labels[currentPointIdx] != 0) {
            continue;
        }
        labels[currentPointIdx] = clusterId;

        int* currentNeighbors = regionQuery(data, currentPointIdx, dataSize, dataDim);
        int currentNeighborCount = dataSize;
        if (currentNeighborCount >= minPoints) {
            for (int i = 0; i < currentNeighborCount; ++i) {
                if (labels[currentNeighbors[i]] == 0 || labels[currentNeighbors[i]] == -1) {
                    pointsQueue.push(currentNeighbors[i]);
                }
            }
        }
        delete[] currentNeighbors;
    }
}

int* DBSCAN::fit(const float* data, int dataSize, int dataDim) {
    int clusterId = 0;
    labels = new int[dataSize];
    for (int i = 0; i < dataSize; ++i) {
        labels[i] = 0;
    }
    
    for (int i = 0; i < dataSize; ++i) {
        if (labels[i] == 0) {
            expandCluster(data, i, ++clusterId, dataSize, dataDim);
        }
        else continue;
    }
    return labels;
}

int main() {
    int m, n;

    m = 100;
    n = 10;
    
    srand(time(NULL));
    
    float data[m][n];
    float* p = data[0];
    
    for (int i = 0; i < m * n; ++i) {
        *(p++) = 1.0 * random() / RAND_MAX;
    }

    DBSCAN dbscan(0.5, 1);
    
    int* labels = dbscan.fit((float*)data, m, n);
    for (int i = 0; i < m; ++i) {
        std::cout << labels[i] << std::endl;
    }

    return 0;
}