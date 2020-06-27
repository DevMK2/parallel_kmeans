#include "kmeans_parallel.cuh"
#include "announce.hh"
#include <algorithm>
#include "log.cc"

static const int UpdateCentroidBlockDim = 1024;

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
__global__
void transposeDataPointers(const DataPoint* const data, Labels_T labels, Trans_DataValues transposed);
__global__
void untransposeDataPointers(const Trans_DataValues transposed, Labels_T labels, DataPoint* const data);

void calcLabelCounts (
    Labels_T const dataLabels,
    Label_T* const dataIDXs,
    size_t* const labelCounts
);

void setLabelBounds (
    const size_t* const labelCounts,
    size_t* const labelFirstIdxes,
    size_t* const labelLastIdxes
);

__global__
void sortDatapoints (
    Labels_T const dataLabels,
    const Label_T* const dataIDXs,
    Trans_DataValues const dataValuesTransposed,
    Trans_DataValues const newDataValuesTransposed
);

void KMeans::main(DataPoint* const centroids, DataPoint* const data) {
#ifdef SPARSE_LOG
    Log<> log ( 
        LogFileName.empty()?  "./results/parallel_mempattern" : LogFileName
    );
#endif

    Labels_T dataLabels = new Label_T[DataSize];
    Trans_DataValues dataValuesTransposed = new Data_T[FeatSize * DataSize];
    Trans_DataValues newDataValuesTransposed;
    cudaAssert (
        cudaHostRegister(dataValuesTransposed, FeatSize*DataSize*sizeof(Data_T), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaMalloc((void**)&newDataValuesTransposed, FeatSize*DataSize*sizeof(Data_T))
    );
    cudaAssert (
        cudaHostRegister(data, DataSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(dataLabels, DataSize*sizeof(Label_T), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(centroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    transposeDataPointers<<<DataSize, FeatSize>>>(data, dataLabels, dataValuesTransposed);

    auto dataIDXs = new int[DataSize]{0,};
    auto labelCounts = new size_t[KSize]{0,};
    auto labelFirstIdxes = new size_t[KSize]{0,};
    auto labelLastIdxes = new size_t[KSize]{0,};
    auto newCentroids = new DataPoint[KSize];
    auto isUpdated = new bool(true);

    cudaAssert (
        cudaHostRegister(dataIDXs, DataSize*sizeof(int), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(newCentroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(isUpdated, sizeof(bool), cudaHostRegisterPortable)
    );

    int numThread_labeling = 256; /*TODO calc from study*/
    int numBlock_labeling = ceil((float)DataSize / numThread_labeling);

#ifdef DEEP_LOG
    Log<LoopEvaluate, 1024> deeplog (
        LogFileName.empty()?  "./results/parallel_mampattern_deep" : LogFileName+"_deep"
    );
#endif
    while(threashold--) {
        cudaDeviceSynchronize();
        memcpyCentroidsToConst(centroids);
        KMeans::labeling<<<numBlock_labeling, numThread_labeling>>>(dataLabels, dataValuesTransposed);
cudaDeviceSynchronize();
cudaAssert( cudaPeekAtLastError());

        cudaDeviceSynchronize();
#ifdef DEEP_LOG
        deeplog.Lap("labeling");
        untransposeDataPointers<<<DataSize, FeatSize>>>(dataValuesTransposed, dataLabels, data);
        cudaDeviceSynchronize();
        announce.Labels(data);
#endif
///////////////////////////////////////////////////////////////
        calcLabelCounts(dataLabels, dataIDXs, labelCounts);
#ifdef DEEP_LOG
        deeplog.Lap("sorting1");
#endif
        setLabelBounds(labelCounts, labelFirstIdxes, labelLastIdxes);
#ifdef DEEP_LOG
        deeplog.Lap("sorting2");
#endif
        memcpyLabelCountToConst(labelCounts, labelFirstIdxes, labelLastIdxes);
#ifdef DEEP_LOG
        deeplog.Lap("sorting3");
#endif
        sortDatapoints<<<numBlock_labeling, numThread_labeling>>> (
            dataLabels, dataIDXs, dataValuesTransposed, newDataValuesTransposed
        );
        cudaMemcpy(dataValuesTransposed, newDataValuesTransposed, DataSize*FeatSize*sizeof(Data_T), cudaMemcpyDeviceToDevice);
cudaDeviceSynchronize();
cudaAssert( cudaPeekAtLastError());
#ifdef DEEP_LOG
        deeplog.Lap("sorting4");
#endif
///////////////////////////////////////////////////////////////
        size_t maxLabelCount = 0;
        for(int i=0; i!=KSize; ++i)
            maxLabelCount = std::max(maxLabelCount, labelCounts[i]);

        resetNewCentroids<<<KSize,FeatSize>>>(newCentroids);
cudaDeviceSynchronize();
cudaAssert( cudaPeekAtLastError());
        dim3 dimBlock(UpdateCentroidBlockDim, 1, 1);
        dim3 dimGrid(ceil((float)maxLabelCount/UpdateCentroidBlockDim), KSize, 1);
        KMeans::updateCentroidAccum<<<dimGrid, dimBlock>>>(newCentroids, dataValuesTransposed);
cudaDeviceSynchronize();
cudaAssert( cudaPeekAtLastError());
        KMeans::updateCentroidDivide<<<KSize, FeatSize>>>(newCentroids);
cudaDeviceSynchronize();
cudaAssert( cudaPeekAtLastError());
#ifdef DEEP_LOG
        cudaDeviceSynchronize();
        deeplog.Lap("updateCentroid");
#endif

        KMeans::checkIsSame<<<KSize, FeatSize>>>(isUpdated, centroids, newCentroids);
        cudaDeviceSynchronize();
        if(*isUpdated)
            break;
        *isUpdated = false;

        memcpyCentroid<<<KSize,FeatSize>>>(centroids, newCentroids);
#ifdef DEEP_LOG
        deeplog.Lap("check centroids");
#endif
    }

    cudaDeviceSynchronize();
    cudaAssert( cudaPeekAtLastError());
#ifdef SPARSE_LOG
    log.Lap("KMeans-Parallel-MemPattern");
#endif
    announce.Labels(data);
    announce.InitCentroids(newCentroids);

    cudaAssert( cudaHostUnregister(data));
    cudaAssert( cudaHostUnregister(dataIDXs));
    cudaAssert( cudaHostUnregister(dataLabels));
    cudaAssert( cudaHostUnregister(dataValuesTransposed));
    cudaAssert( cudaHostUnregister(centroids) );
    cudaAssert( cudaHostUnregister(newCentroids) );
    cudaAssert( cudaHostUnregister(isUpdated) );
    cudaAssert( cudaFree(newDataValuesTransposed));

    delete[] dataIDXs;
    delete[] dataLabels;
    delete[] dataValuesTransposed;
    delete[] labelCounts;
    delete[] labelFirstIdxes;
    delete[] labelLastIdxes;
    delete[] newCentroids;
    delete isUpdated;
}

__global__
void KMeans::labeling(Labels_T const labels, Trans_DataValues const data) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= DataSize)
        return;

    Data_T distSQRSums[KSize]{0,};

    for(int i=0; i!=FeatSize; ++i) {
        Data_T currValue = data[i*DataSize+ idx];

        for(int j=0; j!=KSize; ++j) {
            Data_T currDist = currValue - constCentroidValues[j*FeatSize + i];
            distSQRSums[j] += currDist * currDist;
        }
        __syncthreads();
    }

    Data_T minDistSQRSum = MaxDataValue;
    Label_T minDistLabel = 0;
    for(int i=0; i!=KSize; ++i) {
        if(minDistSQRSum > distSQRSums[i]) {
            minDistSQRSum = distSQRSums[i];
            minDistLabel = i;
        }
    }

    labels[idx] = minDistLabel;
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

    const size_t labelFirstIdx = constLabelFirstIdxes[label]; // TODO Delete me
    const size_t labelLastIdx = constLabelLastIdxes[label];   // TODO Delete me
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

__global__
void transposeDataPointers(const DataPoint* const data, Labels_T labels, Trans_DataValues transposed) {
    int dataIdx = blockIdx.x;
    int valueIdx = threadIdx.x;

    if(valueIdx==0)
        labels[dataIdx] = data[dataIdx].label;

    transposed[valueIdx*DataSize + dataIdx] = data[dataIdx].value[valueIdx];
}

__global__
void untransposeDataPointers(const Trans_DataValues transposed, Labels_T labels, DataPoint* const data) {
    int dataIdx = blockIdx.x;
    int valueIdx = threadIdx.x;

    if(valueIdx==0)
        data[dataIdx].label = labels[dataIdx];

    data[dataIdx].value[valueIdx] = transposed[valueIdx*DataSize + dataIdx];
}

void calcLabelCounts (
    Labels_T const dataLabels,
    Label_T* const dataIDXs,
    size_t* const labelCounts
) {
    memset(labelCounts, 0, KSize*sizeof(size_t));

    for(int i=0; i!=DataSize; ++i) {
        Label_T curr = dataLabels[i];
        dataIDXs[i] = labelCounts[curr];// 라벨 중에 몇 번째인지 index
        labelCounts[curr] += 1; // 라벨 당 datapoint 갯수
    }
}

void setLabelBounds (
    const size_t* const labelCounts,
    size_t* const labelFirstIdxes,
    size_t* const labelLastIdxes
) {
    labelFirstIdxes[0] = 0;
    labelLastIdxes[0] = labelCounts[0] - 1;

    for(int i=1; i!=KSize; ++i) {
        labelFirstIdxes[i] = labelLastIdxes[i-1] + 1;
        labelLastIdxes[i] = labelFirstIdxes[i] + labelCounts[i] - 1;
    }
}

__global__
void sortDatapoints (
    Labels_T const dataLabels,
    const Label_T* const dataIDXs,
    Trans_DataValues const dataValuesTransposed,
    Trans_DataValues const newDataValuesTransposed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= DataSize)
        return;

    int targetIDX = constLabelFirstIdxes[dataLabels[idx]] + dataIDXs[idx];

    for(int j=0; j!=FeatSize; ++j) {
        int row = j*DataSize;
        newDataValuesTransposed[row+targetIDX] = dataValuesTransposed[row+idx];
    }
}