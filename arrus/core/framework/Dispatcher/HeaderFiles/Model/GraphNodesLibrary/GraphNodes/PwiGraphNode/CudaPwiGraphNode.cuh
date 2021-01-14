#ifndef __CUDA_PWI_GRAPH_NODE__
#define __CUDA_PWI_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include <boost/optional.hpp>

enum class PWI_APODIZATION {
    NONE,
    TAS,
    HANN
};

class CudaPwiPixelMap {
public:
    float x, y, width, height;
};

class CudaPwiGraphNode {
private:
    boost::optional <cudaDeviceProp> deviceProp;

    __host__ void loadCudaDeviceProps();

public:
    __host__ void
    pwiRf(const float *inputData, float *output, const cudaStream_t &stream, const int resultWidthInPixels,
          const int resultHeightInPixels, const int anglesCount,
          const float soundVelocity, const float receiverWidth, const int receiversCount, const float samplingFrequency,
          const int samplesCount,
          const float startDepth, const float *anglesInfo, const float *hanningWindow, const CudaPwiPixelMap pixelMap,
          const float transmitFrequency, const PWI_APODIZATION apod);

    __host__ void
    pwiIq(const float2 *inputData, float2 *output, const cudaStream_t &stream, const int resultWidthInPixels,
          const int resultHeightInPixels, const int anglesCount,
          const float soundVelocity, const float receiverWidth, const int receiversCount, const float samplingFrequency,
          const int samplesCount,
          const float startDepth, const float *anglesInfo, const float *hanningWindow, const CudaPwiPixelMap pixelMap,
          const float transmitFrequency, const PWI_APODIZATION apod);
};

#endif