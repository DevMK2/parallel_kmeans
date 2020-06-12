#ifndef __NPY_HH__
#define __NPY_HH__

#include "cnpy.h"
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

using Path_T = std::string;
using Name_T = std::string;
using Shape_T = std::vector<size_t>;

namespace NPY {

template <class T>
inline void saveOverwrite(const Path_T& filePath, const T& data, const Shape_T& shape) {
    cnpy::npy_save(filePath, &data, shape, "w");
}

template <class T>
inline void saveAppend(const Path_T& filePath, const T& data, const Shape_T& shape) {
    cnpy::npy_save(filePath, &data, shape, "a");
}

template <class T>
inline void saveOverwrite(const Path_T& filePath, const std::vector<T>& data, const Shape_T& shape) {
    cnpy::npy_save(filePath, &(data[0]), shape, "w");
}

template <class T>
inline void saveAppend(const Path_T& filePath, const std::vector<T>& data, const Shape_T& shape) {
    cnpy::npy_save(filePath, &(data[0]), shape, "a");
}

template <class T>
inline cnpy::NpyArray load(const Path_T& filePath) {
    return cnpy::npy_load(filePath);
}

template <class T>
inline const T* extract(const cnpy::NpyArray& npyArray) {
    return npyArray.data<T>();
}

}

#endif
