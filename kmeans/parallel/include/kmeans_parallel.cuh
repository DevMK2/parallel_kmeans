#ifndef __KMEANS_PARALLEL_HH__
#define __KMEANS_PARALLEL_HH__

#include <cuda.h>
#include "datatype.hh"
#include "cuda_assert.cuh"
#include "cuda_device.cuh"

namespace KMeans {
void main(DataPoint* const centroids, DataPoint* const data);

///\ No Const, No Sorting
__global__ void labeling(const DataPoint* const centroids, DataPoint* const data);
namespace Labeling {
__device__ Data_T euclideanDistSQR(const Data_T* const lhs, const Data_T* const rhs);
};

///\ Const, No Sorting
__global__ void labeling(DataPoint* const data);
namespace Labeling {
__device__ Data_T euclideanDistSQR(const Data_T* const __restrict__ lhs, const Data_T* const __restrict__ rhs);
};

__global__ void labeling(Labels_T* const labels, Trans_DataValues* const data);

__global__ void updateCentroidAccum(DataPoint* const centroids, const DataPoint* const data);
__global__ void updateCentroidDivide(DataPoint* const centroids);
namespace Update {
__device__ void addValuesLtoR(const Data_T* const lhs, Data_T* const rhs);
};

///\ Transpose
__global__ void labeling(Trans_DataValues const data);
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

__global__ static void memcpyCentroid(DataPoint* const centroids, DataPoint* const newCentroids) {
    centroids[blockIdx.x].value[threadIdx.x]= newCentroids[blockIdx.x].value[threadIdx.x];
}

__global__ static void checkIsSame(bool* const isSame, DataPoint* const centroids, DataPoint* const newCentroids) {
    Data_T prevValue = centroids[blockIdx.x].value[threadIdx.x];
    Data_T newValue = newCentroids[blockIdx.x].value[threadIdx.x];

    if(prevValue - newValue > 0.0001) {
        *isSame = false;
    }
}
};

#endif
