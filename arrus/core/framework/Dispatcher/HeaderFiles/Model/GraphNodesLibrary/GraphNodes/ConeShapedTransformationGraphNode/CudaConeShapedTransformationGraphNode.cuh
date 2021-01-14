#ifndef __CUDA_PHASED_ARRAY_GRAPH_NODE__
#define __CUDA_PHASED_ARRAY_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

class CudaConeShapedTransformationGraphNode {
public:

    __host__ static void coneShapedTransformation(const float *input, float *output, const cudaStream_t &stream,
                                                  const int inputWidthInPixels, const int inputHeightInPixels,
                                                  const float openingAngle, const float outputWidthInMeters,
                                                  const float outputHeightInMeters, const float areaHeight,
                                                  const int outputWidthInPixels, const int outputHeightInPixels);
};

#endif