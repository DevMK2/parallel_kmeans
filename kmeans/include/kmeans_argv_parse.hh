#ifndef __KMENAS_ARGV_PARSE_HH__
#define __KMENAS_ARGV_PARSE_HH__

#include <cassert>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include "npy.hh"
#include "datatype.hh"
#include "announce.hh"

static std::string parseFilePath(const int& argc, const char *argv[]) {
    if(argc < 2 || std::ifstream(argv[1]).bad()) {
        throw std::ifstream::failure("Plz, input arg[1](npy file path) ");
    }
    return std::string(argv[1]);
}

static void arrayToDataPoint(const Data_T* rawData, DataPoint* data) {
    for(int dataIdx=0; dataIdx!=DataSize; ++dataIdx) {
        memcpy((void*)data[dataIdx].value, (void*)rawData, FeatSize*sizeof(Data_T));
        rawData += FeatSize;
    }
}

static void extractDataFromFile(const std::string& filePath, DataPoint* data) {
    /* !! Do not edit below two lines !! */
    auto arr = NPY::load<Data_T>(filePath);
    auto rawData = NPY::extract<Data_T>(arr);
    /* !! cause NPY::load() makes smart pointer which has block scoped lifecycle !! */

    assert(arr.shape.size() == 2);
    assert(DataSize == arr.shape[0]);
    assert(FeatSize == arr.shape[1]);

    announce.DataShapes(arr);

    arrayToDataPoint(rawData, data);
}
#endif