#ifndef __CUDA_HILBERT_GRAPH_NODE__
#define __CUDA_HILBERT_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include <cufft.h>

class CudaHilbertGraphNode {
private:
    int allocatedDataCount;
    cufftHandle cufftPlan;
    cufftComplex *cufftData;

    __host__ void allocMemory(const int batchLength, const int batchCount);

    __host__ void releaseMemory();

public:
    __host__ CudaHilbertGraphNode();

    __host__ ~CudaHilbertGraphNode();

    __host__ void
    hilbertTransform(const float *inputData, float *outputData, const cudaStream_t &stream, const int batchLength,
                     const int batchCount);
};

#endif