#include "kmeans_parallel.cuh"
#include "announce.hh"
#include <algorithm>
//#include <nvrtc.h>

static const int UpdateCentroidBlockDim= 32;

void sortAndGetLabelCounts(DataPoint* const, size_t* const, size_t* const, size_t* const);
void memcpyCentroidsToConst(DataPoint*);
void memcpyCentroidsFromConst(DataPoint*);
void memcpyLabelCountToConst(size_t*, size_t*, size_t*);
void memcpyLabelCountFromConst(size_t*, size_t*, size_t*);

__device__ __constant__ Data_T constCentroidValues[KSize*FeatSize];
__device__ __constant__ size_t constLabelCounts[KSize];
__device__ __constant__ size_t constLabelFirstIdxes[KSize];
__device__ __constant__ size_t constLabelLastIdxes[KSize];


void KMeans::main(DataPoint* const centroids, DataPoint* const data) {
    cudaAssert (
        cudaHostRegister(data, DataSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(centroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );

    auto labelCounts = new size_t[KSize]{0,};
    auto labelFirstIdxes = new size_t[KSize]{0,};
    auto labelLastIdxes = new size_t[KSize]{0,};
    auto newCentroids = new DataPoint[KSize];
    bool* isSame = new bool(true);

    cudaAssert (
        cudaHostRegister(newCentroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(isSame, sizeof(bool), cudaHostRegisterPortable)
    );

    //study(deviceQuery());
    int numThread_labeling = 8; /*TODO get from study*/
    int numBlock_labeling = ceil((float)DataSize / numThread_labeling);

    int threashold = 2;
    while(threashold-- > 0) {
        cudaDeviceSynchronize();
        memcpyCentroidsToConst(centroids);
        KMeans::labeling<<<numBlock_labeling, numThread_labeling>>>(data);

        cudaDeviceSynchronize();
        sortAndGetLabelCounts(data, labelCounts, labelFirstIdxes, labelLastIdxes);
        memcpyLabelCountToConst(labelCounts, labelFirstIdxes, labelLastIdxes);
        size_t maxLabelCount = 0;
        for(int i=0; i!=KSize; ++i)
            maxLabelCount = std::max(maxLabelCount, labelCounts[i]);

        resetNewCentroids<<<KSize,FeatSize>>>(newCentroids);

        dim3 dimBlock(UpdateCentroidBlockDim, 1, 1);
        dim3 dimGrid(ceil(maxLabelCount/UpdateCentroidBlockDim), KSize, 1);
        KMeans::updateCentroidAccum<<<dimGrid, dimBlock>>>(newCentroids, data);
        KMeans::updateCentroidDivide<<<KSize, FeatSize>>>(newCentroids);

        KMeans::checkIsSame<<<KSize, FeatSize>>>(isSame, centroids, newCentroids);
        //cudaDeviceSynchronize();
        //if(isSame)
            //break;

        memcpyCentroid<<<KSize,FeatSize>>>(centroids, newCentroids);
    }

    cudaDeviceSynchronize();
    cudaAssert( cudaPeekAtLastError());
    announce.Labels(data);
    announce.InitCentroids(newCentroids);

    cudaAssert( cudaHostUnregister(data) );
    cudaAssert( cudaHostUnregister(centroids) );
    cudaAssert( cudaHostUnregister(newCentroids) );
    cudaAssert( cudaHostUnregister(isSame) );

    delete[] labelCounts;
    delete[] labelFirstIdxes;
    delete[] labelLastIdxes;
    delete[] newCentroids;
    delete isSame;
}

/// labeling ////////////////////////////////////////////////////////////////////////////////////
__global__
void KMeans::labeling(DataPoint* const data) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= DataSize)
        return;

    DataPoint threadData = data[idx];

    Label_T minDistLabel = 0;
    Data_T minDistSQR = MaxDataValue;

    for(int i=0; i!=KSize; ++i) {
        Data_T currDistSQR = KMeans::Labeling::euclideanDistSQR(threadData.value, constCentroidValues + i*FeatSize);
        if(minDistSQR > currDistSQR) {
            minDistLabel = i;
            minDistSQR = currDistSQR;
        }
    }

    data[idx].label = minDistLabel;
}

__device__ 
Data_T KMeans::Labeling::euclideanDistSQR ( const Data_T* const __restrict__ lhs, const Data_T* const __restrict__ rhs) { 
    const Data_T* valuePtrLHS = lhs;
    const Data_T* valuePtrRHS = rhs;

    Data_T distSQR = 0;

    for(int i=0; i!=FeatSize; ++i) {
        Data_T dist = *valuePtrLHS - *valuePtrRHS;

        distSQR += dist*dist;

        valuePtrLHS++;
        valuePtrRHS++;
    }

    return distSQR;
}

/// update centroids //////////////////////////////////////////////////////////////////////////////
// blockDim = 8~32 ~ 128
// gridDim = 10, ceil(maxLabelCount/blockDim)
// width = maxLabelCount
// blockIdx.y = label
__global__
void KMeans::updateCentroidAccum(DataPoint* const centroids, const DataPoint* const data) {
    __shared__ Data_T Sum[UpdateCentroidBlockDim][FeatSize];

    const int tID = threadIdx.x;
    const Label_T label = blockIdx.y;

    const size_t labelFirstIdx = constLabelFirstIdxes[label]; // TODO Const mem으로 보내기
    const size_t labelLastIdx = constLabelLastIdxes[label]; // TODO Const mem으로 보내기
    const size_t dataIdx = labelFirstIdx + (blockIdx.x * blockDim.x + tID);

    if(dataIdx > labelLastIdx)
        return;

    for(int i=0; i!=FeatSize; ++i)
        Sum[tID][i] = data[dataIdx].value[i];
    __syncthreads();// TODO 없어도 되나?

    {//\Asserts
    assert(label >= 0 && label < 10);
    assert(labelLastIdx < DataSize);
    assert(data[dataIdx].label == label);
    }

    for(int stride=blockDim.x/2; stride>=1; stride>>=1) {
        if(tID < stride && dataIdx+stride <= labelLastIdx)
            Update::addValuesLtoR(Sum[tID+stride], Sum[tID]);
        __syncthreads();
    }

    if(tID != 0)
        return;

    for(int i=0; i!=FeatSize; ++i)
        atomicAdd(&(centroids[label].value[i]), Sum[tID][i]);
}

__global__
void KMeans::updateCentroidDivide(DataPoint* const centroids) {
    int label = blockIdx.x;
    centroids[label].value[threadIdx.x] /= constLabelCounts[label];
}

__device__
void KMeans::Update::addValuesLtoR(const Data_T* const lhs, Data_T* const rhs) {
    const Data_T* lhsPtr = lhs;
    Data_T* rhsPtr = rhs;

    for(int i=0; i!=FeatSize; ++i)
        *(rhsPtr++) += *(lhsPtr++);
}

void sortAndGetLabelCounts (
    DataPoint* const data,
    size_t* const labelCounts,
    size_t* const labelFirstIdxes,
    size_t* const labelLastIdxes
) {
    std::sort(data, data+DataSize, cmpDataPoint);

    const DataPoint* dataPtr = data;

    Label_T currLabel = dataPtr->label;
    int currLabelCount = 0;
    labelFirstIdxes[currLabelCount] = 0;

    for(int i=0; i!=DataSize; ++i) {
        if(currLabel != dataPtr->label) {
            labelCounts[currLabel] = currLabelCount;
            labelLastIdxes[currLabel] = i-1;

            currLabelCount = 0;
            currLabel = dataPtr->label;
            labelFirstIdxes[currLabel] = i;
        }
        currLabelCount++;
        dataPtr++;
    }
    labelCounts[currLabel] = currLabelCount;
    labelLastIdxes[currLabel] = DataSize-1;
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

void memcpyLabelCountToConst(size_t* labelCount, size_t* labelFirstIdxes, size_t* labelLastIdxes) {
    cudaAssert (
        cudaMemcpyToSymbol(constLabelCounts, labelCount, KSize*sizeof(size_t))
    );
    cudaAssert (
        cudaMemcpyToSymbol(constLabelFirstIdxes, labelFirstIdxes, KSize*sizeof(size_t))
    );
    cudaAssert (
        cudaMemcpyToSymbol(constLabelLastIdxes, labelLastIdxes, KSize*sizeof(size_t))
    );
}

void memcpyLabelCountFromConst(size_t* labelCount, size_t* labelFirstIdxes, size_t* labelLastIdxes) {
    cudaAssert (
        cudaMemcpyFromSymbol(labelCount, constLabelCounts, KSize*sizeof(size_t))
    );
    cudaAssert (
        cudaMemcpyFromSymbol(labelFirstIdxes, constLabelFirstIdxes, KSize*sizeof(size_t))
    );
    cudaAssert (
        cudaMemcpyFromSymbol(labelLastIdxes, constLabelLastIdxes, KSize*sizeof(size_t))
    );
}