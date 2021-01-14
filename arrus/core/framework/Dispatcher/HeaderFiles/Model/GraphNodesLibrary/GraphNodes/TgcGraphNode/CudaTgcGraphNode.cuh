#ifndef __CUDA_PWI_GRAPH_NODE__
#define __CUDA_PWI_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include <unordered_map>

enum class TGC_FUNC_TYPE {
    LIN,
    EXP
};

class CudaTgcGraphNode {
public:
    __host__ static void
    tgc(const float *inputData, float *output, const cudaStream_t &stream, const int width, const int height,
        const float areaHeight,
        const std::unordered_map<std::string, float> &params, const TGC_FUNC_TYPE tgcFuncType);
};

#endif