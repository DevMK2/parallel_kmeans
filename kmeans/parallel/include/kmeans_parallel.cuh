#ifndef __KMEANS_PARALLEL_CUH__
#define __KMEANS_PARALLEL_CUH__

#include <cuda.h>
#include "datatype.hh"
#include "data_transpose.cuh"
#include "cuda_assert.cuh"
#include "cuda_device.cuh"

namespace KMeans {
void main(DataPoint* const centroids, DataPoint* const data);

///\ No Const, No Sorting
__global__ void labeling(const DataPoint* const centroids, DataPoint* const data);

//TODO deprecated
namespace Labeling {
__device__ Data_T euclideanDistSQR(const Data_T* const lhs, const Data_T* const rhs);
};

///\ Const, No Sorting
__global__ void labeling(DataPoint* const data);
namespace Labeling {
__device__ Data_T euclideanDistSQR(const Data_T* const __restrict__ lhs, const Data_T* const __restrict__ rhs);
};

__global__ void labeling(Labels_T const labels, Trans_DataValues const data);

__global__ void updateCentroidAccum(DataPoint* const centroids, const DataPoint* const data);
__global__ void updateCentroidDivide(DataPoint* const centroids);
namespace Update {
__device__ void addValuesLtoR(const Data_T* const lhs, Data_T* const rhs);
};

///\ Transpose
__global__ void labeling(Trans_DataValues const data);
__device__ void calcAndSetDistSQRSums(const int& dataIDX, Trans_DataValues const data, Data_T* const distSQRSums);
__device__ Label_T getMinDistLabel(const Data_T* const distSQRSums);

void calcLabelCounts(Labels_T const dataLabels, Label_T* const dataIDXs, size_t* const labelCounts);
void setLabelBounds(const size_t* const labelCounts, size_t* const labelFirstIdxes, size_t* const labelLastIdxes);
__global__
void sortDatapoints (
    Labels_T const dataLabels,
    const Label_T* const dataIDXs,
    Trans_DataValues const dataValuesTransposed,
    Trans_DataValues const newDataValuesTransposed
);

__global__ void updateCentroidAccum(DataPoint* const centroids, const Trans_DataValues data);


static void initCentroids(DataPoint* const centroids, const DataPoint* const data) {
    for(int kIdx=0; kIdx!=KSize; ++kIdx) {
        centroids[kIdx].label = kIdx;

        for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
            centroids[kIdx].value[featIdx] = data[kIdx].value[featIdx];
        }
    }
}

__global__ static void resetNewCentroids(DataPoint* newCentroids) {
    newCentroids[blockIdx.x].label = 0;
    newCentroids[blockIdx.x].value[threadIdx.x]= 0;
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

__global__ static void memcpyCentroid(DataPoint* const centroids, DataPoint* const newCentroids) {
    centroids[blockIdx.x].value[threadIdx.x]= newCentroids[blockIdx.x].value[threadIdx.x];
}

};

#endif
