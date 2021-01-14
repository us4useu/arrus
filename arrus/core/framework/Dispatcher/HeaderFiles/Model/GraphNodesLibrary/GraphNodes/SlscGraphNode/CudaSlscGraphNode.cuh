#ifndef __CUDA_STA_GRAPH_NODE__
#define __CUDA_STA_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include <boost/optional.hpp>

class CudaSlscPixelMap {
public:
    float x, y, width, height;
};

class CudaSlscGraphNode {
private:
    boost::optional <cudaDeviceProp> deviceProp;

    __host__ void loadCudaDeviceProps();

    __host__ void
    removeConstantRf(float *inputPtr, const int samplesCount, const int signalsCount, const cudaStream_t &stream);

    __host__ void
    removeConstantIq(float2 *inputPtr, const int samplesCount, const int signalsCount, const cudaStream_t &stream);

public:
    __host__ void
    slscRf(float *inputPtr, float *outputPtr, const cudaStream_t &stream, const int width, const int height,
           const float areaHeight,
           const int receiversCount, const int samplesCount, const float receiverWidth, const float soundVelocity,
           const float samplingFrequency,
           const float startDepth, const int *transmittersIndexes, const int transmittersCount, const int lags,
           const int offset, const CudaSlscPixelMap pixelMap);

    __host__ void
    slscIq(float2 *inputPtr, float *outputPtr, const cudaStream_t &stream, const int width, const int height,
           const float areaHeight,
           const int receiversCount, const int samplesCount, const float receiverWidth, const float soundVelocity,
           const float samplingFrequency,
           const float startDepth, const int *transmittersIndexes, const int transmittersCount, const int lags,
           const float transmitFrequency, const CudaSlscPixelMap pixelMap);

    __host__ void
    slscWithFocusingRf(float *inputPtr, float *outputPtr, const cudaStream_t &stream, const int width, const int height,
                       const float areaHeight,
                       const int receiversCount, const int samplesCount, const float receiverWidth,
                       const float soundVelocity,
                       const float samplingFrequency, const float startDepth, const int lags, const int offset);

    __host__ void slscWithFocusingIq(float2 *inputPtr, float *outputPtr, const cudaStream_t &stream, const int width,
                                     const int height, const float areaHeight,
                                     const int receiversCount, const int samplesCount, const float receiverWidth,
                                     const float soundVelocity,
                                     const float samplingFrequency, const float startDepth, const int lags,
                                     const float transmitFrequency);
};

#endif