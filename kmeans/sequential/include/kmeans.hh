#ifndef __KMEANS_HH__
#define __KMEANS_HH__

#include "datatype.hh"
#include <string.h>

namespace KMeans {
void main(DataPoint* const centroids, DataPoint* const data);

static void initCentroids(DataPoint* const centroids, const DataPoint* const data) {
    for(int kIdx=0; kIdx!=KSize; ++kIdx) {
        centroids[kIdx].label = kIdx;

        for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
            centroids[kIdx].value[featIdx] = data[kIdx].value[featIdx];
        }
    }
}

void labeling(const DataPoint* const centroids, DataPoint* const data);
namespace Labeling {
Data_T euclideanDistSQR(const DataPoint* const lhs, const DataPoint* const rhs);
void setClosestCentroid(const DataPoint* centroids, DataPoint* const data);
};

void updateCentroid(DataPoint* const centroids, const DataPoint* const data);
namespace Update {
    void addValuesLtoR(const Data_T* const lhs, Data_T* const rhs);
};

static bool isSame(DataPoint* const centroids, DataPoint* const newCentroids) {
    DataPoint* prevCentroidPtr = centroids;
    DataPoint* newCentroidPtr = newCentroids;

    for(int kIdx=0; kIdx!=KSize; ++kIdx) {
        Data_T* prevValuePtr = prevCentroidPtr->value;
        Data_T* newValuePtr = newCentroidPtr->value;

        for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
            if(*prevValuePtr - *newValuePtr > 0.0001) {
                return false;
            }

            prevValuePtr++;
            newValuePtr++;
        }

        prevCentroidPtr++;
        newCentroidPtr++;
    }
    return true;
}
};

static void resetNewCentroids(DataPoint* newCentroids) {
    for(int i=0; i!=KSize; ++i) {
        newCentroids[i].label = i;
        Data_T* valuePtr = newCentroids[i].value;
        
        for(int j=0; j!=FeatSize; ++j) {
            *valuePtr = 0.0;
            valuePtr++;
        }
    }
}

static inline void memcpyCentroid(DataPoint* const centroids, DataPoint* const newCentroids) {
    memcpy((void*)centroids, (void*)newCentroids, KSize*sizeof(DataPoint));
}

#endif
