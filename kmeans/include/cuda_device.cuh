/***********************************************/
/*                                             */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*                                             */
/***********************************************/
/*************************************************************/
/*                                                           */
/*                                                           */
/* This code came from NVIDIA-CUDA_Samples of 'device_query' */
/*                                                           */
/*                                                           */
/*************************************************************/

#ifndef __CUDA_SMCORES_HH__
#define __CUDA_SMCORES_HH__
#include <vector>
#include <cuda_runtime.h>
#include "cuda_assert.cuh"

static int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128},
        {0x53, 128}, {0x60,  64}, {0x61, 128}, {0x62, 128}, {0x70,  64}, {0x72,  64},
        {0x75,  64}, {-1, -1}
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    return nGpuArchCoresPerSM[index - 1].Cores;
}

using Count_T = unsigned int;
using MemSize_L = unsigned long;
using MemSize_LL = unsigned long long;

struct DeviceQuery {
    Count_T index;
    Count_T multiProcessors;
    Count_T cudaCores;
    Count_T numRegPerBlock;
    Count_T threadsPerBlock;
    Count_T threadsPerMultiprocesser;
    MemSize_L shmPerBlock;
    MemSize_L totalConstMem;
    MemSize_LL totalGlobalMem;
};

static std::vector<DeviceQuery> deviceQuery() {
    int deviceCount = 0;
    cudaAssert(cudaGetDeviceCount(&deviceCount));

    auto queries = std::vector<DeviceQuery>(deviceCount);

    for (int i=0; i!=deviceCount; ++i) {
        cudaSetDevice(i);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        DeviceQuery query;

        query.index = i;
        query.multiProcessors = deviceProp.multiProcessorCount;
        query.cudaCores = query.multiProcessors * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        query.numRegPerBlock = deviceProp.regsPerBlock;
        query.threadsPerBlock = deviceProp.maxThreadsPerBlock;
        query.threadsPerMultiprocesser = deviceProp.maxThreadsPerMultiProcessor;
        query.shmPerBlock = deviceProp.sharedMemPerBlock;
        query.totalConstMem = deviceProp.totalConstMem;
        query.totalGlobalMem = (unsigned long long)deviceProp.totalGlobalMem;

        queries[i] = query;
    }

    return queries;
}

#endif
