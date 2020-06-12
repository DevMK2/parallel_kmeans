#ifndef __CUDA_ASSERT__
#define __CUDA_ASSERT__
#include <stdio.h>

#define cudaAssert(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code == cudaSuccess) return;

    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}

#endif
