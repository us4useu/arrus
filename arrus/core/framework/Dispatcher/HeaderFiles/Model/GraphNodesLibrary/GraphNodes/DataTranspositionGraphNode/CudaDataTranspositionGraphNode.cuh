#ifndef __CUDA_DATA_TRANSPOSITION_GRAPH_NODE__
#define __CUDA_DATA_TRANSPOSITION_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

class CudaDataTranspositionGraphNode {
public:
    __host__ static void transposeData(const short *inputPtr, float *outputPtr, const cudaStream_t &stream,
                                       const int xSize, const int ySize, const int zSize);

    __host__ static void transposeData(const short *inputPtr, short *outputPtr, const cudaStream_t &stream,
                                       const int xSize, const int ySize, const int zSize);

    __host__ static void transposeData(const int *inputPtr, int *outputPtr, const cudaStream_t &stream,
                                       const int xSize, const int ySize, const int zSize);

    __host__ static void transposeData(const float *inputPtr, float *outputPtr, const cudaStream_t &stream,
                                       const int xSize, const int ySize, const int zSize);

    __host__ static void transposeData(const double *inputPtr, double *outputPtr, const cudaStream_t &stream,
                                       const int xSize, const int ySize, const int zSize);

    __host__ static void transposeData(const float2 *inputPtr, float2 *outputPtr, const cudaStream_t &stream,
                                       const int xSize, const int ySize, const int zSize);
};

#endif