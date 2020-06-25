#include "announce.hh"
#include "kmeans.hh"
#include "log.cc"

void KMeans::main(DataPoint* const centroids, DataPoint* const data) {
#ifdef SPARSE_LOG
    Log<> log ( 
        LogFileName.empty()?  "./results/sequential" : LogFileName
    );
#endif
    auto newCentroids = new DataPoint[KSize];

#ifdef DEEP_LOG
    Log<LoopEvaluate, 1024> deeplog (
        LogFileName.empty()?  "./results/sequential_deep" : LogFileName+"_deep"
    );
#endif
    while(threashold--) {
        KMeans::labeling(centroids, data);
        announce.Labels(data);
#ifdef DEEP_LOG
        deeplog.Lap("labeling");
#endif 

        resetNewCentroids(newCentroids);
        KMeans::updateCentroid(newCentroids, data);
#ifdef DEEP_LOG
        deeplog.Lap("updateCentroid");
#endif 

        if(KMeans::isSame(centroids, newCentroids))
            break;

        memcpyCentroid(centroids, newCentroids);
#ifdef DEEP_LOG
        deeplog.Lap("check centroids");
#endif 
    }

    delete[] newCentroids;

#ifdef SPARSE_LOG
    log.Lap("KMeans-Sequential End");
#endif
}

/// labeling ////////////////////////////////////////////////////////////////////////////////////

void KMeans::labeling(const DataPoint* const centroids, DataPoint* const data) {
    DataPoint* dataPtr = data;

    for(int dataIdx=0; dataIdx!=DataSize; ++dataIdx)
        Labeling::setClosestCentroid(centroids, dataPtr++);
}

void KMeans::Labeling::setClosestCentroid( const DataPoint* centroids, DataPoint* const data) {
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

void KMeans::updateCentroid(DataPoint* const centroids, const DataPoint* const data) {
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

void KMeans::Update::addValuesLtoR(const Data_T* const lhs, Data_T* const rhs) {
    const Data_T* lhsPtr = lhs;
    Data_T* rhsPtr = rhs;

    for(int featIdx=0; featIdx!=FeatSize; ++featIdx)
        *(rhsPtr++) += *(lhsPtr++);
}
