#include "kmeans_parallel.cuh"
#include "announce.hh"

void KMeans::main(DataPoint* const centroids, DataPoint* const data) {
    Announce announce(KSize, DataSize, FeatSize);

    cudaAssert (
        cudaHostRegister(data, DataSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );
    cudaAssert (
        cudaHostRegister(centroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );

    auto newCentroids = new DataPoint[KSize];
    cudaAssert (
        cudaHostRegister(newCentroids, KSize*sizeof(DataPoint), cudaHostRegisterPortable)
    );

    bool* isSame = new bool;
    *isSame = true;
    cudaAssert (
        cudaHostRegister(isSame, sizeof(bool), cudaHostRegisterPortable)
    );

    //study(deviceQuery());
    int numThread_labeling = 128; /*TODO get from study*/
    int numBlock_labeling = ceil((float)DataSize / numThread_labeling);

    int threashold = 30; // 
    while(threashold-- > 0) {
        KMeans::labeling<<<numBlock_labeling, numThread_labeling>>>(centroids, data);

        resetNewCentroids<<<KSize,FeatSize>>>(newCentroids);

        KMeans::updateCentroidAccum<<<numBlock_labeling,numThread_labeling>>>(newCentroids, data);
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

    cudaAssert( cudaHostUnregister(data) );
    cudaAssert( cudaHostUnregister(centroids) );
    cudaAssert( cudaHostUnregister(newCentroids) );
    cudaAssert( cudaHostUnregister(isSame) );

    delete[] data;
    delete[] centroids;
    delete[] newCentroids;
    delete isSame;
}

/// initCentroids /////////////////////////////////////////////////////////////////////////////

void KMeans::initCentroids(DataPoint* const centroids, const DataPoint* const data) {
    for(int kIdx=0; kIdx!=KSize; ++kIdx) {
        centroids[kIdx].label = kIdx;

        for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
            centroids[kIdx].value[featIdx] = data[kIdx].value[featIdx];
        }
    }
}

/// labeling ////////////////////////////////////////////////////////////////////////////////////
__global__
void KMeans::labeling(const DataPoint* const centroids, DataPoint* const data) {
    const int& idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= DataSize)
        return;
    Labeling::setClosestCentroid(centroids, data+idx);
}

__device__
void KMeans::Labeling::setClosestCentroid(const DataPoint* centroids, DataPoint* const data) {
    const DataPoint* centroidPtr = centroids;
    size_t minDistLabel = 0;
    Data_T minDistSQR = MaxDataValue;

    for(int kIdx=0; kIdx!=KSize; ++kIdx) {
        Data_T currDistSQR = euclideanDistSQR(centroidPtr, data);

        if(minDistSQR > currDistSQR) {
            minDistLabel = centroidPtr->label;
            minDistSQR = currDistSQR;
        }

        centroidPtr++;
    }

    data->label = minDistLabel;
}

__device__
Data_T KMeans::Labeling::euclideanDistSQR ( const DataPoint* const lhs, const DataPoint* const rhs) {
    const Data_T* valuePtrLHS = lhs->value;
    const Data_T* valuePtrRHS = rhs->value;

    Data_T distSQR = 0;

    for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
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

__global__
void KMeans::checkIsSame(bool* const isSame, DataPoint* const centroids, DataPoint* const newCentroids) {
    Data_T prevValue = centroids[blockIdx.x].value[threadIdx.x];
    Data_T newValue = newCentroids[blockIdx.x].value[threadIdx.x];

    if(prevValue != newValue) {
        *isSame = false;
    }
}

__global__
void resetNewCentroids(DataPoint* newCentroids) {
    newCentroids[blockIdx.x].label = 0;
    newCentroids[blockIdx.x].value[threadIdx.x]= 0;
}

__global__
void memcpyCentroid(DataPoint* const centroids, DataPoint* const newCentroids) {
    centroids[blockIdx.x].value[threadIdx.x]= newCentroids[blockIdx.x].value[threadIdx.x];
}

void study(const std::vector<DeviceQuery>& devices) {
    /*
     * According to the CUDA C Best Practice Guide.
     * 1. Thread per block should be a multiple of 32(warp size)
     * 2. A minimum of 64 threads per block should be used.
     * 3. Between 128 and 256 thread per block is a better choice
     * 4. Use several(3 to 4) small thread blocks rather than one large thread block
     */
    /* 
     * sizeof DataPoint 
     *   = 4(float) * 200(feature size) + 4(label, int) 
     *   = 804 byte
     *   =>register memory per thread
     *     = 832 byte { 804 + 8(pointer) + 8(two int) + 8(size_t) + 4(Data_T) }
     *   =>register count per thread
     *     = 832/4 = 208
     *
     * sizeof Centroid
     *   = DataPoint x 10
     *   = 8040 byte
     * 
     * memory per block (* NOT SHARED MEMORY *)
     *   = 804 * 64 
     *   = 51456 byte
     *
     * total global memory size = 8112 MBytes
     * number of registers per block = 65536
     */
    Count_T numRegisterPerKernel_labeling = 208;
    MemSize_L sizeDataPoint = sizeof(DataPoint);
    MemSize_L sizeCentroids = sizeDataPoint * KSize;
    for(auto device : devices) {
        assert(sizeCentroids < device.totalConstMem);

        std::cout <<  "Device["<<device.index<<"]" << std::endl;

        Count_T maxThreadsPerBlock = device.numRegPerBlock / numRegisterPerKernel_labeling;
        std::cout <<"max threads per block(labeling) : " << maxThreadsPerBlock << std::endl;
        std::cout <<"max threads per block(update)   : " << maxThreadsPerBlock << std::endl;
        std::cout <<"max threads per block(check)    : " << maxThreadsPerBlock << std::endl;

        std::cout << device.numRegPerBlock / 208.0 << std::endl;
        std::cout << device.threadsPerBlock << std::endl;
        std::cout << device.threadsPerMultiprocesser << std::endl;
    }
}

void Sequential::updateCentroid(DataPoint* const centroids, const DataPoint* const data) {
    int labelSizes[KSize] = {0,}; 
    
    // 모든 데이터 포인트의 값을 해당하는 centroid에 더한다.
    const DataPoint* dataPtr = data;
    for(int dataIdx=0; dataIdx!=DataSize; ++dataIdx) {
        labelSizes[dataPtr->label]++;

        Update::addValuesLtoR(dataPtr->value, centroids[dataPtr->label].value);
        dataPtr++;
    }

    DataPoint* centroidPtr = centroids;
    for(int kIdx=0; kIdx!=KSize; ++kIdx) {
        int labelSize = labelSizes[centroidPtr->label];
        Data_T* valuePtr = centroidPtr->value;

        for(int featIdx=0; featIdx!=FeatSize; ++featIdx) {
            *(valuePtr++) /= labelSize;
        }
        centroidPtr++;
    }
}

void Sequential::Update::addValuesLtoR(const Data_T* const lhs, Data_T* const rhs) {
    const Data_T* lhsPtr = lhs;
    Data_T* rhsPtr = rhs;

    for(int featIdx=0; featIdx!=FeatSize; ++featIdx)
        *(rhsPtr++) += *(lhsPtr++);
}
