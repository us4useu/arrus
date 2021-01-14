#ifndef __CUDA_QUADRATURE_DEMODULATION_GRAPH_NODE__
#define __CUDA_QUADRATURE_DEMODULATION_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

class CudaQuadratureDemodulationGraphNode {
public:
    __host__ static void
    rfToIq(const float *inputRfPtr, float2 *outputIqPtr, const cudaStream_t &stream, const int batchLength,
           const int batchCount, const float samplingFrequency, const float transmitFrequency,
           const int startSampleNumber);

    __host__ static void
    decimation(const float2 *inputIqPtr, float2 *outputIqPtr, const cudaStream_t &stream, const int batchLength,
               const int batchCount, const int decimationValue);
};

#endif