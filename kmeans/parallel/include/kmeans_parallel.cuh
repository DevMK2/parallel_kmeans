#ifndef __KMEANS_PARALLEL_HH__
#define __KMEANS_PARALLEL_HH__

#include "datatype.hh"
#include <cuda.h>
#include "cuda_assert.cuh"
#include "cuda_device.cuh"

void study(const std::vector<DeviceQuery>& devices);
inline void resetNewCentroids(DataPoint* newCentroids);
inline void memcpyCentroid(DataPoint* const centroids, DataPoint* const newCentroids);

namespace KMeans {
void main(DataPoint* const centroids, DataPoint* const data);

void initCentroids(DataPoint* const centroids, const DataPoint* const data);

__global__ void labeling(const DataPoint* const centroids, DataPoint* const data);
namespace Labeling {
__device__ Data_T euclideanDistSQR(const DataPoint* const lhs, const DataPoint* const rhs);
__device__ void setClosestCentroid(const DataPoint* centroids, DataPoint* const data);
};

void updateCentroid(DataPoint* const centroids, const DataPoint* const data);
namespace Update {
    void addValuesLtoR(const Data_T* const lhs, Data_T* const rhs);
};

bool isSame(DataPoint* const centroids, DataPoint* const newCentroids);
};

#endif
