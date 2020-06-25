#include "kmeans_parallel.cuh"
#include "announce.hh"
#include "log.cc"

void memcpyCentroidsToConst(DataPoint* centroids);
void memcpyCentroidsFromConst(DataPoint* centroids);

__device__ __constant__ Data_T constCentroidValues[KSize*FeatSize];

void KMeans::main(DataPoint* const centroids, DataPoint* const data) {
#ifdef SPARSE_LOG
    Log<> log ( 
        LogFileName.empty()?  "./results/parallel_const" : LogFileName
    );
#endif
    cudaAssert (
        cudaHostRegister(data, DataSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(centroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );

    auto newCentroids = new DataPoint[KSize];
    bool* isUpdated = new bool(true);

    cudaAssert (
        cudaHostRegister(newCentroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(isUpdated, sizeof(bool), cudaHostRegisterPortable)
    );

    //study(deviceQuery());
    int numThread_labeling = 8; /*TODO get from study*/
    int numBlock_labeling = ceil((float)DataSize / numThread_labeling);

#ifdef DEEP_LOG
    Log<LoopEvaluate, 1024> deeplog (
        LogFileName.empty()?  "./results/parallel_const_deep" : LogFileName+"_deep"
    );
#endif
    while(threashold--) {
        cudaDeviceSynchronize();
        memcpyCentroidsToConst(centroids);
        KMeans::labeling<<<numBlock_labeling, numThread_labeling>>>(data);
#ifdef DEEP_LOG
        cudaDeviceSynchronize();
        announce.Labels(data);
        deeplog.Lap("labeling");
#endif

        resetNewCentroids<<<KSize,FeatSize>>>(newCentroids);

        KMeans::updateCentroidAccum<<<numBlock_labeling,numThread_labeling>>>(newCentroids, data);
        KMeans::updateCentroidDivide<<<KSize, FeatSize>>>(newCentroids);
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
    log.Lap("KMeans-Parallel-const End");
#endif
    announce.Labels(data);
    announce.InitCentroids(newCentroids);

    cudaAssert( cudaHostUnregister(data) );
    cudaAssert( cudaHostUnregister(centroids) );
    cudaAssert( cudaHostUnregister(newCentroids) );
    cudaAssert( cudaHostUnregister(isUpdated) );

    delete[] newCentroids;
    delete isUpdated;
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
__global__
void KMeans::updateCentroidAccum(DataPoint* const centroids, const DataPoint* const data) {
    const int dataIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(dataIdx >= DataSize)
        return;

    const int centroidIdx = data[dataIdx].label;

    atomicAdd(&(centroids[centroidIdx].label), 1); // newCentroids는 labelSize를 나타내기 위해 0으로 초기화됨
    Update::addValuesLtoR(data[dataIdx].value, centroids[centroidIdx].value);
}

__global__
void KMeans::updateCentroidDivide(DataPoint* const centroids) {
    centroids[blockIdx.x].value[threadIdx.x] /= centroids[blockIdx.x].label;
}

__device__
void KMeans::Update::addValuesLtoR(const Data_T* const lhs, Data_T* const rhs) {
    const Data_T* lhsPtr = lhs;
    Data_T* rhsPtr = rhs;

    for(int featIdx=0; featIdx!=FeatSize; ++featIdx)
        atomicAdd(rhsPtr++, *(lhsPtr++));
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
