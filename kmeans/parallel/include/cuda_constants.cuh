#ifndef __CUDA_CONSTANTS_CUH__
#define __CUDA_CONSTANTS_CUH__

#include <cuda.h>
#include "datatype.hh"
#include "data_transpose.cuh"
#include "cuda_assert.cuh"

__device__ __constant__ Data_T constCentroidValues[KSize*FeatSize];
__device__ __constant__ size_t constLabelFirstIdxes[KSize];
__device__ __constant__ size_t constLabelLastIdxes[KSize];

void memcpyLabelCountToConst(size_t* labelFirstIdxes, size_t* labelLastIdxes) {
    cudaAssert (
        cudaMemcpyToSymbol(constLabelFirstIdxes, labelFirstIdxes, KSize*sizeof(size_t))
    );
    cudaAssert (
        cudaMemcpyToSymbol(constLabelLastIdxes, labelLastIdxes, KSize*sizeof(size_t))
    );
}

void memcpyLabelCountFromConst(size_t* labelFirstIdxes, size_t* labelLastIdxes) {
    cudaAssert (
        cudaMemcpyFromSymbol(labelFirstIdxes, constLabelFirstIdxes, KSize*sizeof(size_t))
    );
    cudaAssert (
        cudaMemcpyFromSymbol(labelLastIdxes, constLabelLastIdxes, KSize*sizeof(size_t))
    );
}

void memcpyCentroidsToConst(DataPoint* centroids) {
    Data_T values[KSize*FeatSize];
    
    for(int i=0; i!=KSize; ++i) {
        for(int j=0; j!=FeatSize; ++j) {
            values[i*FeatSize+j] = centroids[i].value[j];
        }
    }
    cudaAssert (
        cudaMemcpyToSymbol(constCentroidValues, values, KSize*FeatSize*sizeof(Data_T))
    );
}

void memcpyCentroidsFromConst(DataPoint* centroids) {
    Label_T labels[KSize];
    Data_T values[KSize*FeatSize];

    cudaAssert (
        cudaMemcpyFromSymbol(values, constCentroidValues, KSize*FeatSize*sizeof(Data_T))
    );

    for(int i=0; i!=KSize; ++i) {
        centroids[i].label = labels[i];

        for(int j=0; j!=FeatSize; ++j) {
            centroids[i].value[j] = values[i*FeatSize+j];
        }
    }
}
#endif