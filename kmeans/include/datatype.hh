#ifndef __DATATYPE_HH__
#define __DATATYPE_HH__
#include "npy.hh" // npy size_t
#include "config.hh"
#include <vector>

using Label_T = int;
using Data_T = float;
using Data_V = std::vector<Data_T>;

struct DataPoint {
    Label_T label = 0;
    Data_T value[FeatSize];
};

inline bool cmpDataPoint(const DataPoint& p1, const DataPoint& p2) {
    return p1.label < p2.label;
}

#endif
