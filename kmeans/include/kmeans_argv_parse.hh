#ifndef __KMENAS_ARGV_PARSE_HH__
#define __KMENAS_ARGV_PARSE_HH__

#include <cassert>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include "npy.hh"
#include "config.hh"
#include "datatype.hh"
#include "announce.hh"

static std::string parseFilePath(const int& argc, const char *argv[]) {
    if(argc < 2 || std::ifstream(argv[1]).fail()) {
        if(std::ifstream(DefaultInputFile).fail()){
            throw std::ifstream::failure (
                "Can't find input files in arg[1](npy file path) OR default input file ("
                + DefaultInputFile+ ")"
            );
        }

        std::cerr << "You have no such file : " 
            << argv[1]
            << " so, I'm running with default data file : "
            << DefaultInputFile 
            << std::endl;

        return DefaultInputFile;
    }
    return std::string(argv[1]);
}

static void arrayToDataPoint(const Data_T* rawData, DataPoint* data) {
    for(int i=0; i!=DATA_SCALE; ++i) {

        int begin = i*ORIGIN_DATA_SIZE, end = (i+1)*ORIGIN_DATA_SIZE;
        const Data_T* rawDataPtr = rawData;

        for(int dataIdx=begin; dataIdx!=end; ++dataIdx) {
            memcpy((void*)data[dataIdx].value, (void*)rawDataPtr, FeatSize*sizeof(Data_T));
            rawDataPtr += FeatSize;
        }
    }
}

static void extractDataFromFile(const std::string& filePath, DataPoint* data) {
    /* !! Do not edit below two lines !! */
    auto arr = NPY::load<Data_T>(filePath);
    auto rawData = NPY::extract<Data_T>(arr);
    /* !! cause NPY::load() makes smart pointer which has block scoped lifecycle !! */

    assert(arr.shape.size() == 2);
    assert(ORIGIN_DATA_SIZE == arr.shape[0]);
    assert(FeatSize == arr.shape[1]);

    announce.DataShapes(arr);

    arrayToDataPoint(rawData, data);
}

#endif
