#ifndef __CUDA_PHASED_ARRAY_GRAPH_NODE__
#define __CUDA_PHASED_ARRAY_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

class CudaPhasedArrayGraphNode {
public:

    __host__ static void phasedArray(const float *input, float *output, const cudaStream_t &stream,
                                     const int resultWidthInPixels, const int resultHeightInPixels,
                                     const float soundVelocity, const float areaHeight,
                                     const float receiverWidth, const int receiversCount, const float samplingFrequency,
                                     const int samplesCount,
                                     const float openingAngle, const float2 *focusesInfo, const int focusesNumber);

    __host__ static void phasedArrayWithFocusing(const float *input, float *output, const cudaStream_t &stream,
                                                 const int resultWidthInPixels, const int resultHeightInPixels,
                                                 const float soundVelocity, const float areaHeight,
                                                 const float receiverWidth, const int receiversCount,
                                                 const float samplingFrequency, const int samplesCount,
                                                 const float *transmitAngles);
};

#endif