#include "kmeans_parallel.cuh"
#include "announce.hh"
#include <algorithm>
#include "log.cc"

static const int UpdateCentroidBlockDim = 256;

void sortAndGetLabelCounts(DataPoint* const, size_t* const, size_t* const, size_t* const);
void memcpyCentroidsToConst(DataPoint*);
void memcpyCentroidsFromConst(DataPoint*);
void memcpyLabelCountToConst(size_t*, size_t*, size_t*);
void memcpyLabelCountFromConst(size_t*, size_t*, size_t*);

__device__ __constant__ Data_T constCentroidValues[KSize*FeatSize];
__device__ __constant__ size_t constLabelCounts[KSize];
__device__ __constant__ size_t constLabelFirstIdxes[KSize];
__device__ __constant__ size_t constLabelLastIdxes[KSize];

#define Trans_DataValues_IDX(x,y) y*DataSize+x
#define CentroidValues_IDX(x,y) y*FeatSize+x

Labels_T dataLabels;
Trans_DataValues dataValuesTransposed;

void transposDataPointers(const DataPoint* const data, Trans_DataValues transposed) {
    for(int i=0; i!=DataSize; ++i) {
        dataLabels[i] = data[i].label;

        for(int j=0; j!=FeatSize; ++j) {
            transposed[j*DataSize + i] = data[i].value[j];
        }
    }

    // TODO Delete me
    for(int i=0; i!=FeatSize; ++i)
        for(int j=0; j!=DataSize; ++j)
            assert(transposed[i*DataSize + j] == data[j].value[i]);
}

void untransposDataPointers(const Trans_DataValues transposed, DataPoint* const data) {
    for(int i=0; i!=DataSize; ++i) {
        data[i].label = dataLabels[i];

        for(int j=0; j!=FeatSize; ++j) {
            data[i].value[j] = transposed[j*DataSize + i];
        }
    }
}

void KMeans::main(DataPoint* const centroids, DataPoint* const data) {
    Log<> log("./results/parallel");
    transposDataPointers(data, dataValuesTransposed);

    cudaAssert (
        cudaHostRegister(dataLabels, DataSize*sizeof(Label_T), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(dataValuesTransposed, FeatSize*DataSize*sizeof(Data_T), cudaHostRegisterPortable)
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
    int numThread_labeling = 512; /*TODO get from study*/
    int numBlock_labeling = ceil((float)DataSize / numThread_labeling);

    int threashold = 20;
    while(threashold-- > 0) {
        cudaDeviceSynchronize();
        memcpyCentroidsToConst(centroids);
        KMeans::labeling<<<numBlock_labeling, numThread_labeling>>>(&dataLabels, &dataValuesTransposed);

        cudaDeviceSynchronize();
        untransposDataPointers(dataValuesTransposed, data);
        sortAndGetLabelCounts(data, labelCounts, labelFirstIdxes, labelLastIdxes);
        transposDataPointers(data, dataValuesTransposed);
        announce.Labels(data);

        memcpyLabelCountToConst(labelCounts, labelFirstIdxes, labelLastIdxes);
        size_t maxLabelCount = 0;
        for(int i=0; i!=KSize; ++i)
            maxLabelCount = std::max(maxLabelCount, labelCounts[i]);

        resetNewCentroids<<<KSize,FeatSize>>>(newCentroids);

        dim3 dimBlock(UpdateCentroidBlockDim, 1, 1);
        dim3 dimGrid(ceil(maxLabelCount/UpdateCentroidBlockDim), KSize, 1);
        KMeans::updateCentroidAccum<<<dimGrid, dimBlock>>>(newCentroids, dataValuesTransposed);
        KMeans::updateCentroidDivide<<<KSize, FeatSize>>>(newCentroids);

        //announce.InitCentroids(newCentroids);

        KMeans::checkIsSame<<<KSize, FeatSize>>>(isSame, centroids, newCentroids);
        cudaDeviceSynchronize();
        if(*isSame)
            break;
        *isSame = true;

        memcpyCentroid<<<KSize,FeatSize>>>(centroids, newCentroids);
    }

    cudaDeviceSynchronize();
    cudaAssert( cudaPeekAtLastError());
    announce.Labels(data);
    announce.InitCentroids(newCentroids);

    cudaAssert( cudaHostUnregister(dataLabels));
    cudaAssert( cudaHostUnregister(dataValuesTransposed));
    cudaAssert( cudaHostUnregister(centroids) );
    cudaAssert( cudaHostUnregister(newCentroids) );
    cudaAssert( cudaHostUnregister(isSame) );

    delete[] labelCounts;
    delete[] labelFirstIdxes;
    delete[] labelLastIdxes;
    delete[] newCentroids;
    delete isSame;
    log.Lap("KMeans-Parallel End");
}

/// labeling ////////////////////////////////////////////////////////////////////////////////////
__global__
void KMeans::labeling(Labels_T* const labels, Trans_DataValues* const data) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= DataSize)
        return;

    Data_T distSQRSums[KSize]{0,};

    for(int i=0; i!=FeatSize; ++i) {
        Data_T currValue = (*data)[Trans_DataValues_IDX(idx, i)];

        for(int j=0; j!=KSize; ++j) {
            Data_T currDist = currValue - constCentroidValues[CentroidValues_IDX(i,j)];
            distSQRSums[j] += currDist * currDist;
        }
    }

    Data_T minDistSQRSum = MaxDataValue;
    Label_T minDistLabel = 0;
    for(int i=0; i!=KSize; ++i) {
        if(minDistSQRSum > distSQRSums[i]) {
            minDistSQRSum = distSQRSums[i];
            minDistLabel = i;
        }
    }

    (*labels)[idx]= minDistLabel;
}

/// update centroids //////////////////////////////////////////////////////////////////////////////
// blockDim = 8~32 ~ 128
// gridDim = 10, ceil(maxLabelCount/blockDim)
// width = maxLabelCount
// blockIdx.y = label
__global__
void KMeans::updateCentroidAccum(DataPoint* const centroids, const Trans_DataValues data) {
    __shared__ Data_T Sum[UpdateCentroidBlockDim];

    const int tID = threadIdx.x;
    const Label_T label = blockIdx.y;

    const size_t labelFirstIdx = constLabelFirstIdxes[label];
    const size_t labelLastIdx = constLabelLastIdxes[label];
    const size_t dataIdx = labelFirstIdx + (blockIdx.x * blockDim.x + tID);

    if(dataIdx > labelLastIdx)
        return;

    {//\Asserts
    assert(label >= 0 && label < 10);
    assert(labelLastIdx < DataSize);
    assert(dataIdx >= labelFirstIdx);
    }

    for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
        Sum[tID] = data[Trans_DataValues_IDX(dataIdx, featIdx)];
        __syncthreads();// TODO 없어도 되나?

        for(int stride=blockDim.x/2; stride>=1; stride>>=1) {
            if(tID < stride && dataIdx+stride <= labelLastIdx)
                Sum[tID] += Sum[tID+stride];
            __syncthreads();
        }

        if(tID != 0)
            continue;

        atomicAdd(&(centroids[label].value[featIdx]), Sum[tID]);
    }
}

__global__
void KMeans::updateCentroidDivide(DataPoint* const centroids) {
    int label = blockIdx.x;
    centroids[label].value[threadIdx.x] /= constLabelCounts[label];
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
