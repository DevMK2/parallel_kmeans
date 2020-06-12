#include <stdlib.h>
#include <string.h>
#include <cmath>

#include "npy.hh"
#include "datatype.hh"
#include "announce.hh"
#include "log.cc"

#include <fstream>

#include "kmeans_parallel.cuh"

void arrayToDataPoint(const Data_T* rawData, DataPoint* data);

int main(int argc, const char *argv[]) {
    if(argc < 2 || std::ifstream(argv[1]).bad()) {
        fprintf(stderr, "Plz, input arg[1](npy file path) ");
        return -1;
    }

    /* !! Do not edit below two lines !! */
    auto arr = NPY::load<Data_T>(argv[1]);
    auto rawData = NPY::extract<Data_T>(arr);
    /* !! cause NPY::load() makes smart pointer which has block scoped lifecycle !! */

    assert(arr.shape.size() == 2);
    assert(DataSize == arr.shape[0]);
    assert(FeatSize == arr.shape[1]);

    Announce announce(KSize, DataSize, FeatSize);
    announce.DataShapes(arr);

    auto data = new DataPoint[DataSize];
    arrayToDataPoint(rawData, data);

    auto centroids = new DataPoint[KSize];
    KMeans::initCentroids(centroids, data);
    announce.InitCentroids(centroids);

    {
    Log<> log("./log");
    KMeans::main(centroids, data);
    log.Lap("KMeans End");
    }

    return 0;
}

void arrayToDataPoint(const Data_T* rawData, DataPoint* data) {
    for(int dataIdx=0; dataIdx!=DataSize; ++dataIdx) {
        memcpy((void*)data[dataIdx].value, (void*)rawData, FeatSize*sizeof(Data_T));
        rawData += FeatSize;
    }
}
