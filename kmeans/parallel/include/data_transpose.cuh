#ifndef __DATA_TRANSPOSE_CUH__
#define __DATA_TRANSPOSE_CUH__

#include "datatype.hh"

void transposeDataPointers(const DataPoint* const data, Label_T* labels, Data_T* transposed) {
    for(int i=0; i!=DataSize; ++i) {
        for(int j=0; j!=FeatSize; ++j) {
            transposed[j*DataSize + i] = data[i].value[j];
        }
    }
}

void untransposeDataPointers(DataPoint* const data, Label_T* labels, Data_T* transposed) {
    for(int i=0; i!=DataSize; ++i) {
        for(int j=0; j!=FeatSize; ++j) {
            data[i].value[j] = transposed[j*DataSize + i];
        }
    }
}

#endif