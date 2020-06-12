#ifndef __ANNOUNCE_HH__
#define __ANNOUNCE_HH__

#include "stdio.h"
#include <iostream>
#include "datatype.hh"

class Announce {
private:
    size_t KSize, DataSize, FeatSize; 

public:
    Announce(const size_t& kSize, const size_t& dataSize, const size_t& featSize)
    : KSize(kSize), DataSize(dataSize), FeatSize(featSize) {
        std::cout << "\nAnnounce constructed..\n" << std::endl;
    }

    void DataShapes(const cnpy::NpyArray& npArray) {
        printf("Byte size of feature value: %ld\n", npArray.word_size);
        printf("K: %ld, Features: %ld, Datas: %ld\n\n", KSize, FeatSize, DataSize);
    }

    void InitCentroids(const DataPoint* centroid) {
        static int maxVisibleLength = 6;
        std::cout << "initialized centroids .. " << std::endl;

        for(int kIdx=0; kIdx!=KSize; ++kIdx) {
            std::cout << centroid[kIdx].label << ":[ " ;

            for(int featIdx=0; featIdx!=maxVisibleLength; ++featIdx)
                printf("%-7.3f ", centroid[kIdx].value[featIdx]);

            std::cout << "...]" << std::endl;
        }
        std::cout << std::endl;
    }

    void Labels(const DataPoint* data) {
        static int tryCount = 0;
        int labelSizes[KSize] = {0,}; 

        std::cout << "labeling result, number of datas foreach label .. (try:"<<++tryCount<< ")" << std::endl;
        for(int dataIdx=0; dataIdx!=DataSize; ++dataIdx)
            labelSizes[data[dataIdx].label]++;

        for(int kIdx=0; kIdx!=KSize; ++kIdx)
            std::cout << kIdx << " : " << labelSizes[kIdx] << std::endl;

        std::cout << std::endl;
    }
};

#endif
