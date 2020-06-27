#ifndef __KMEANS_PARALLEL_CUH__
#define __KMEANS_PARALLEL_CUH__

#include <cuda.h>
#include "datatype.hh"
#include "data_transpose.cuh"
#include "cuda_device.cuh"

namespace KMeans {
void main(DataPoint* const centroids, DataPoint* const data);

__global__ void labeling(Label_T* const labels, Data_T* const data);
__device__ void calcAndSetDistSQRSums(const int& dataIDX, Data_T* const data, Data_T* const distSQRSums);
__device__ Label_T getMinDistLabel(const Data_T* const distSQRSums);

void calcLabelCounts(const Label_T* const dataLabels, Label_T* const dataIDXs, size_t* const labelCounts);
void setLabelBounds(const size_t* const labelCounts, size_t* const labelFirstIdxes, size_t* const labelLastIdxes);
__global__ void sortDatapoints (const Label_T* const, const Label_T* const, const Data_T* const, Data_T* const);

__global__ void updateCentroidAccum(DataPoint* const centroids, const Data_T* data);
__global__ void updateCentroidDivide(DataPoint* const centroids);

__global__ static void resetNewCentroids(DataPoint* newCentroids) {
    newCentroids[blockIdx.x].label = 0;
    newCentroids[blockIdx.x].value[threadIdx.x]= 0;
}

__global__ static void memcpyCentroid(DataPoint* const centroids, DataPoint* const newCentroids) {
    centroids[blockIdx.x].value[threadIdx.x]= newCentroids[blockIdx.x].value[threadIdx.x];
}

static void initCentroids(DataPoint* const centroids, const DataPoint* const data) {
    for(int kIdx=0; kIdx!=KSize; ++kIdx) {
        centroids[kIdx].label = kIdx;

        for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
            centroids[kIdx].value[featIdx] = data[kIdx].value[featIdx];
        }
    }
}

static bool isConvergence(DataPoint* const centroids, DataPoint* const newCentroids) {
    DataPoint* prevCentroidPtr = centroids;
    DataPoint* newCentroidPtr = newCentroids;
    for(int kIdx=0; kIdx!=KSize; ++kIdx) {
        Data_T* prevValuePtr = prevCentroidPtr->value;
        Data_T* newValuePtr = newCentroidPtr->value;

        for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
            if(*prevValuePtr - *newValuePtr > 0.0001)
                return false;
            prevValuePtr++; newValuePtr++;
        }
        prevCentroidPtr++; newCentroidPtr++;
    }
    return true;
}
};

#endif
