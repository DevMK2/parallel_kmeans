#ifndef __DATA_TRANSPOSE_CUH__
#define __DATA_TRANSPOSE_CUH__

#include "datatype.hh"

using Labels_T = Label_T*;
using Trans_DataValues = Data_T*;

#define Trans_DataValues_IDX(x,y) y*DataSize+x
#define CentroidValues_IDX(x,y) y*FeatSize+x

void transposeDataPointers(const DataPoint* const data, Labels_T labels, Trans_DataValues transposed) {
    for(int i=0; i!=DataSize; ++i) {
        for(int j=0; j!=FeatSize; ++j) {
            transposed[j*DataSize + i] = data[i].value[j];
        }
    }
}

void untransposeDataPointers(DataPoint* const data, Labels_T labels, Trans_DataValues transposed) {
    for(int i=0; i!=DataSize; ++i) {
        for(int j=0; j!=FeatSize; ++j) {
            data[i].value[j] = transposed[j*DataSize + i];
        }
    }
}

#endif