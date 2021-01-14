#ifndef __CUDA_BMODE_GRAPH_NODE__
#define __CUDA_BMODE_GRAPH_NODE__

#include <cuda_runtime.h>
#include <cfloat>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"


class CudaBModeGraphNode {
private:
//		CudaReductionUtils minCudaReductionUtils, maxCudaReductionUtils;

public:
    __host__ void
    convertToBMode(const float *inputPtr, float *outputPtr, const cudaStream_t &stream, const float minDBLimit,
                   const float maxDBLimit, const int dataCount,
                   const float maxDataValue = FLT_MAX);

    __host__ void
    convertToBModeIq(const float2 *inputPtr, float *complexModulusPtr, float *outputPtr, const cudaStream_t &stream,
                     const float minDBLimit, const float maxDBLimit, const int dataCount,
                     const float maxDataValue = FLT_MAX);
};

#endif
