#ifndef __KMEANS_PARALLEL_HH__
#define __KMEANS_PARALLEL_HH__

#include "datatype.hh"
#include <cuda.h>
#include "cuda_assert.cuh"
#include "cuda_device.cuh"

namespace KMeans {
void main(DataPoint* const centroids, DataPoint* const data);

void initCentroids(DataPoint* const centroids, const DataPoint* const data);

__global__ void labeling(const DataPoint* const centroids, DataPoint* const data);
namespace Labeling {
    __device__ Data_T euclideanDistSQR(const DataPoint* const lhs, const DataPoint* const rhs);
    __device__ void setClosestCentroid(const DataPoint* centroids, DataPoint* const data);
};

__global__
void updateCentroidAccum(DataPoint* const centroids, const DataPoint* const data);
__global__
void updateCentroidDivide(DataPoint* const centroids);
namespace Update {
    __device__
    void addValuesLtoR(const Data_T* const lhs, Data_T* const rhs);
};

__global__
void checkIsSame(bool* const isSame, DataPoint* const centroids, DataPoint* const newCentroids);
};

__global__
void resetNewCentroids(DataPoint* newCentroids);

__global__
void memcpyCentroid(DataPoint* const centroids, DataPoint* const newCentroids);

namespace Sequential{ 
void updateCentroid(DataPoint* const centroids, const DataPoint* const data);
namespace Update {
    void addValuesLtoR(const Data_T* const lhs, Data_T* const rhs);
};
};

#endif
