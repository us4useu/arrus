#ifndef __CUDA_STA_GRAPH_NODE__
#define __CUDA_STA_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include <boost/optional.hpp>

enum class STA_APODIZATION {
    NONE,
    TAS,
    HANN
};

class CudaStaPixelMap {
public:
    float x, y, width, height;
};

class CudaStaGraphNode {
private:
    boost::optional <cudaDeviceProp> deviceProp;

    __host__ void loadCudaDeviceProps();

public:
    __host__ void
    staRf(const float *inputPtr, float *outputPtr, cudaStream_t &stream, const int width, const int height,
          const int receiversCount, const int samplesCount, const float receiverWidth, const float soundVelocity,
          const float samplingFrequency,
          const float startDepth, const float *transmittersIndexes, const int transmittersCount,
          const float *hanningWindow, const float transmitFrequency,
          const STA_APODIZATION apod, const CudaStaPixelMap pixelMap);

    __host__ void
    staIq(const float2 *inputPtr, float *outputPtr, cudaStream_t &stream, const int width, const int height,
          const int receiversCount, const int samplesCount, const float receiverWidth, const float soundVelocity,
          const float samplingFrequency,
          const float startDepth, const float *transmittersIndexes, const int transmittersCount,
          const float *hanningWindow, const float transmitFrequency,
          const STA_APODIZATION apod, const CudaStaPixelMap pixelMap);

    __host__ void
    staRfWithFocusing(const float *inputPtr, float *outputPtr, cudaStream_t &stream, const int width, const int height,
                      const float areaHeight,
                      const int receiversCount, const int samplesCount, const float receiverWidth,
                      const float soundVelocity, const float samplingFrequency,
                      const float startDepth, const float *hanningWindow, const float transmitFrequency,
                      const STA_APODIZATION apod);

    __host__ void
    staIqWithFocusing(const float2 *inputPtr, float *outputPtr, cudaStream_t &stream, const int width, const int height,
                      const float areaHeight,
                      const int receiversCount, const int samplesCount, const float receiverWidth,
                      const float soundVelocity, const float samplingFrequency,
                      const float startDepth, const float *hanningWindow, const float transmitFrequency,
                      const STA_APODIZATION apod);
};

#endif