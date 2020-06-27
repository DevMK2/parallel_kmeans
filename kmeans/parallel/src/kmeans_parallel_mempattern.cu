#include "kmeans_parallel.cuh"
#include "cuda_constants.cuh"
#include "announce.hh"
#include "log.cc"

static const int labelingThreads = 256; // TODO get from study
static const int updateCentroidThreads = 1024;

void KMeans::main(DataPoint* const centroids, DataPoint* const data) {
#ifdef SPARSE_LOG
    Log<> log ( 
        LogFileName.empty()?  "./results/parallel_mempattern" : LogFileName
    );
#endif
    //\ memory access pattern을 개선하기 위해 row major하도록 datapoints를 전치함.
    Labels_T dataLabels = new Label_T[DataSize];
    Trans_DataValues dataValuesTransposed = new Data_T[FeatSize * DataSize];
    transposeDataPointers(data, dataLabels, dataValuesTransposed);
    cudaAssert (
        cudaHostRegister(dataLabels, DataSize*sizeof(Label_T), cudaHostRegisterPortable)
    );
    cudaAssert ( //TODO pinned할 필요 없을 듯 
        cudaHostRegister(dataValuesTransposed, FeatSize*DataSize*sizeof(Data_T), cudaHostRegisterPortable)
    );

    //\ datapoints 재배치 위한 추가 data
    Trans_DataValues newDataValuesTransposed;
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

    auto labelCounts = new size_t[KSize]{0,};
    auto labelFirstIdxes = new size_t[KSize]{0,};
    auto labelLastIdxes = new size_t[KSize]{0,};
    auto dataIdxs = new int[DataSize]{0,};
    cudaAssert (
        cudaHostRegister(dataIdxs, DataSize*sizeof(int), cudaHostRegisterPortable)
    );

#ifdef DEEP_LOG
    Log<LoopEvaluate, 1024> deeplog (
        LogFileName.empty()?  "./results/parallel_mampattern_deep" : LogFileName+"_deep"
    );
#endif
    dim3 dimBlockLabeling(labelingThreads);// TODO calc from study
    dim3 dimGridLabeling(ceil((float)DataSize / labelingThreads));

    while(threashold--) {
        memcpyCentroidsToConst(centroids);
        KMeans::labeling<<<dimGridLabeling, dimBlockLabeling>>>(dataLabels, dataValuesTransposed);
#ifdef DEEP_LOG
        deeplog.Lap("labeling");
#endif

        cudaDeviceSynchronize(); // labeling에서 갱신된 dataLabels로 label당 datapoint의 갯수를 세야 한다.
        KMeans::calcLabelCounts(dataLabels, dataIdxs, labelCounts);
        KMeans::setLabelBounds(labelCounts, labelFirstIdxes, labelLastIdxes);
        memcpyLabelCountToConst(labelFirstIdxes, labelLastIdxes);
        KMeans::sortDatapoints<<<dimGridLabeling, dimBlockLabeling>>> (
            dataLabels, dataIdxs, dataValuesTransposed, newDataValuesTransposed
        );
        cudaMemcpy(dataValuesTransposed, newDataValuesTransposed, DataSize*FeatSize*sizeof(Data_T), cudaMemcpyDeviceToDevice);
#ifdef DEEP_LOG
        deeplog.Lap("sorting");
#endif

        KMeans::resetNewCentroids<<<KSize,FeatSize>>>(newCentroids);
#ifdef DEEP_LOG
        cudaDeviceSynchronize();
        deeplog.Lap("reset NewCentroids");
#endif

        //\ TODO 스트리밍 분석 다시하자
        size_t maxLabelCount = 0;
        for(int i=0; i!=KSize; ++i)
            maxLabelCount = std::max(maxLabelCount, labelCounts[i]);
        dim3 dimBlockUpdateCentroid(updateCentroidThreads, 1, 1);
        dim3 dimGridUpdateCentroid(ceil((float)maxLabelCount/updateCentroidThreads), KSize, 1);
        KMeans::updateCentroidAccum<<<dimGridUpdateCentroid, dimBlockUpdateCentroid>>> (
            newCentroids, dataValuesTransposed
        );
        KMeans::updateCentroidDivide<<<KSize, FeatSize>>>(newCentroids);
#ifdef DEEP_LOG
        cudaDeviceSynchronize();
        deeplog.Lap("updateCentroid");
#endif

        if(isConvergence(centroids, newCentroids))
            break;
        memcpyCentroid<<<KSize,FeatSize>>>(centroids, newCentroids);
        cudaDeviceSynchronize(); // 다음 루프에서 갱신된 centroids를 constant mem으로 올려야한다.
#ifdef DEEP_LOG
        deeplog.Lap("check centroids");
#endif
    }

    cudaDeviceSynchronize();
    cudaAssert( cudaPeekAtLastError());
#ifdef SPARSE_LOG
    log.Lap("KMeans-Parallel-MemPattern");
#endif
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
    delete[] labelFirstIdxes;
    delete[] labelLastIdxes;
    delete[] newCentroids;
}

//// Labeling //////////////////////////////////////////////////////////////////////////////////////
__global__
void KMeans::labeling(Labels_T const labels, Trans_DataValues const data) {
    const int dataIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(dataIdx >= DataSize)
        return;

    Data_T distSQRSums[KSize]{0,};
    calcAndSetDistSQRSums(dataIdx, data, distSQRSums);

    labels[dataIdx] = getMinDistLabel(distSQRSums);
}

__device__
void KMeans::calcAndSetDistSQRSums(const int& dataIdx, Trans_DataValues const data, Data_T* const distSQRSums) {
    for(int i=0; i!=FeatSize; ++i) {
        int featIdx = i * DataSize;

        Data_T currValue = data[dataIdx + featIdx];

        for(int j=0; j!=KSize; ++j) {
            Data_T currDist = currValue - constCentroidValues[j*FeatSize + i];
            distSQRSums[j] += currDist * currDist;
        }
        __syncthreads(); // TODO memory access 패턴을 위해 동기화. 이것도 streaming으로 필 수 있을 듯
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
void KMeans::calcLabelCounts(Labels_T const dataLabels, Label_T* const dataIdxs, size_t* const labelCounts) {
    memset(labelCounts, 0, KSize*sizeof(size_t));

    for(int i=0; i!=DataSize; ++i) {
        Label_T curr = dataLabels[i];
        dataIdxs[i] = labelCounts[curr];// 라벨 중에 몇 번째인지 index
        labelCounts[curr] += 1; // 라벨 당 datapoint 갯수
    }
}

void KMeans::setLabelBounds (
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
void KMeans::sortDatapoints (
    Labels_T const dataLabels,
    const Label_T* const dataIdxs,
    Trans_DataValues const dataValuesTransposed,
    Trans_DataValues const newDataValuesTransposed
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
void KMeans::updateCentroidAccum(DataPoint* const centroids, const Trans_DataValues data) {
    __shared__ Data_T Sum[updateCentroidThreads];

    const int tID = threadIdx.x;
    const Label_T label = blockIdx.y;

    const size_t labelFirstIdx = constLabelFirstIdxes[label]; // TODO Delete me
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
        __syncthreads();

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
    int labelCount = constLabelLastIdxes[label] - constLabelFirstIdxes[label] + 1;
    centroids[label].value[threadIdx.x] /= labelCount;
}