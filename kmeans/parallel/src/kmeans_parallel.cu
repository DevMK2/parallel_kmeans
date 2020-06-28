#include "kmeans_parallel.cuh"
#include "cuda_constants.cuh"
#include "cuda_assert.cuh"
#include "announce.hh"

static const int labelingThreads = 256;
static const int updateCentroidThreads = 1024;

void KMeans::main(DataPoint* const centroids, DataPoint* const data) {
    //\ memory access pattern을 개선하기 위해 row major하도록 datapoints를 전치함.
    Label_T* dataLabels = new Label_T[DataSize];
    Data_T* dataValuesTransposed = new Data_T[FeatSize * DataSize];
    transposeDataPointers(data, dataLabels, dataValuesTransposed);
    cudaAssert (
        cudaHostRegister(dataLabels, DataSize*sizeof(Label_T), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(dataValuesTransposed, FeatSize*DataSize*sizeof(Data_T), cudaHostRegisterPortable)
    );

    //\ datapoints 재배치 위한 추가 data
    auto labelCounts = new size_t[KSize]{0,};
    auto begines = new size_t[KSize]{0,};
    auto endes = new size_t[KSize]{0,};
    auto dataIdxs = new int[DataSize]{0,};
    Data_T* newDataValuesTransposed;
    cudaAssert (
        cudaHostRegister(dataIdxs, DataSize*sizeof(int), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaMalloc((void**)&newDataValuesTransposed, FeatSize*DataSize*sizeof(Data_T))
    );

    //\ update된 centroid와 기존 centroid를 비교하기 위한 추가 data
    auto newCentroids = new DataPoint[KSize];
    cudaAssert (
        cudaHostRegister(centroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(newCentroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );

    dim3 dimBlockLabeling(labelingThreads);
    dim3 dimGridLabeling(ceil((float)DataSize / labelingThreads));

    while(threashold--) {
        announce.Loop("KMeans Parallel");

        //\ Labeling
        memcpyCentroidsToConst(centroids);
        KMeans::labeling<<<dimGridLabeling, dimBlockLabeling>>>(dataLabels, dataValuesTransposed);

        //\ Realignment 
        cudaDeviceSynchronize(); // labeling에서 갱신된 dataLabels로 label당 datapoint의 갯수를 세야 한다.
        KMeans::calcLabelCounts(dataLabels, dataIdxs, labelCounts);
        KMeans::setLabelBounds(labelCounts, begines, endes);
        memcpyLabelCountToConst(begines, endes);

        KMeans::sortDatapoints<<<dimGridLabeling, dimBlockLabeling>>> (
            dataLabels, dataIdxs, dataValuesTransposed, newDataValuesTransposed
        );
        cudaMemcpy(dataValuesTransposed, newDataValuesTransposed, DataSize*FeatSize*sizeof(Data_T), cudaMemcpyDeviceToDevice);

        KMeans::resetNewCentroids<<<KSize,FeatSize>>>(newCentroids);

        //\ UpdateCentroids
        size_t maxLabelCount = 0;
        for(int i=0; i!=KSize; ++i)
            maxLabelCount = std::max(maxLabelCount, labelCounts[i]);
        dim3 dimBlockUpdateCentroid(updateCentroidThreads, 1, 1);
        dim3 dimGridUpdateCentroid(ceil((float)maxLabelCount/updateCentroidThreads), KSize, 1);
        KMeans::updateCentroidAccum<<<dimGridUpdateCentroid, dimBlockUpdateCentroid>>> (
            newCentroids, dataValuesTransposed
        );
        KMeans::updateCentroidDivide<<<KSize, FeatSize>>>(newCentroids);

        //\ Check is convergnece
        if(isConvergence(centroids, newCentroids))
            break;
        memcpyCentroid<<<KSize,FeatSize>>>(centroids, newCentroids);
        cudaDeviceSynchronize(); // 다음 루프에서 갱신된 centroids를 constant mem으로 올려야한다.
    }

    cudaDeviceSynchronize();
    cudaAssert( cudaPeekAtLastError());
    announce.Centroids(newCentroids);

    cudaAssert( cudaHostUnregister(dataIdxs));
    cudaAssert( cudaHostUnregister(dataLabels));
    cudaAssert( cudaHostUnregister(dataValuesTransposed));
    cudaAssert( cudaHostUnregister(centroids) );
    cudaAssert( cudaHostUnregister(newCentroids) );
    cudaAssert( cudaFree(newDataValuesTransposed));

    delete[] dataIdxs;
    delete[] dataLabels;
    delete[] dataValuesTransposed;
    delete[] labelCounts;
    delete[] begines;
    delete[] endes;
    delete[] newCentroids;
}

//// Labeling //////////////////////////////////////////////////////////////////////////////////////
__global__
void KMeans::labeling(Label_T* const labels, Data_T* const data) {
    const int dataIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(dataIdx >= DataSize)
        return;

    Data_T distSQRSums[KSize]{0,};
    calcAndSetDistSQRSums(dataIdx, data, distSQRSums);

    labels[dataIdx] = getMinDistLabel(distSQRSums);
}

__device__
void KMeans::calcAndSetDistSQRSums(const int& dataIdx, Data_T* const data, Data_T* const distSQRSums) {
    for(int i=0; i!=FeatSize; ++i) {
        int featIdx = i * DataSize;
        Data_T currValue = data[dataIdx + featIdx];

        for(int j=0; j!=KSize; ++j) {
            Data_T centroidValue = constCentroidValues[j*FeatSize + i];
            Data_T currDist = currValue - centroidValue;
            distSQRSums[j] += currDist * currDist;
        }
        __syncthreads();
    }
}

__device__
Label_T KMeans::getMinDistLabel(const Data_T* const distSQRSums) {
    const Data_T* distSQRSumPtr = distSQRSums;

    Data_T minDistSQRSum = MaxDataValue;
    Label_T minDistLabel = 0;

    for(int i=0; i!=KSize; ++i) {
        if(minDistSQRSum > *distSQRSumPtr) {
            minDistSQRSum = *distSQRSumPtr;
            minDistLabel = i;
        }
        distSQRSumPtr++;
    }

    return minDistLabel;
}

/// realign datapoints ///////////////////////////////////////////////////////////////////////////
void KMeans::calcLabelCounts (
    const Label_T* const dataLabels,
    Label_T* const dataIdxs,
    size_t* const labelCounts
) {
    memset(labelCounts, 0, KSize*sizeof(size_t));

    for(int i=0; i!=DataSize; ++i) {
        Label_T curr = dataLabels[i];
        dataIdxs[i] = labelCounts[curr]; // 라벨 중에 몇 번째인지 index
        labelCounts[curr] += 1; // 라벨 당 datapoint 갯수
    }
}

void KMeans::setLabelBounds (
    const size_t* const labelCounts,
    size_t* const begines,
    size_t* const endes
) {
    begines[0] = 0;
    endes[0] = labelCounts[0] - 1;

    for(int i=1; i!=KSize; ++i) {
        begines[i] = endes[i-1] + 1;
        endes[i] = begines[i] + labelCounts[i] - 1;
    }
}

__global__
void KMeans::sortDatapoints (
    const Label_T* const dataLabels,
    const Label_T* const dataIdxs,
    const Data_T* const dataValuesTransposed,
    Data_T* const newDataValuesTransposed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= DataSize)
        return;

    int targetIdx = constLabelFirstIdxes[dataLabels[idx]] + dataIdxs[idx];

    for(int j=0; j!=FeatSize; ++j) {
        int row = j*DataSize;
        newDataValuesTransposed[row+targetIdx] = dataValuesTransposed[row+idx];
    }
}

/// update centroids //////////////////////////////////////////////////////////////////////////////
__global__
void KMeans::updateCentroidAccum(DataPoint* const centroids, const Data_T* data) {
    __shared__ Data_T Sum[updateCentroidThreads];

    const int tID = threadIdx.x;
    const Label_T label = blockIdx.y;

    const size_t begin = constLabelFirstIdxes[label];
    const size_t end = constLabelLastIdxes[label];
    const size_t dataIdx = begin + (blockIdx.x * blockDim.x + tID);

    if(dataIdx > end)
        return;

    {//\Asserts
    assert(label >= 0 && label < KSize);
    assert(end < DataSize);
    assert(dataIdx >= begin);
    }

    for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
        Sum[tID] = data[featIdx*DataSize + dataIdx];

        for(int stride=blockDim.x/2; stride>=1; stride>>=1) {
            __syncthreads();
            if(tID < stride && dataIdx+stride <= end)
                Sum[tID] += Sum[tID+stride];
        }

        if(tID != 0)
            continue;

        atomicAdd(&(centroids[label].value[featIdx]), Sum[tID]);
    }
}

__global__
void KMeans::updateCentroidDivide(DataPoint* const centroids) {
    int label = blockIdx.x;
    int labelCount = constLabelLastIdxes[label] - constLabelFirstIdxes[label] + 1;
    centroids[label].value[threadIdx.x] /= labelCount;
}