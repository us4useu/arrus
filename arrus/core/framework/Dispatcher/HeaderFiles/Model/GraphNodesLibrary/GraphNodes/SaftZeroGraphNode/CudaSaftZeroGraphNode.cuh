#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

class CudaSaftZeroGraphNode {
private:
    int allocatedApertureLen;
    int allocatedSignalLen;
    int fftApertureLen;
    int fftSignalLen;
    float *devFkz;
    cufftHandle cufftPlan;
    cufftComplex *cufftData;
    cufftComplex *cufftOutData;

    __host__ void allocStructures(const int apertureLen,
                                  const int signalLen,
                                  const float soundVelocity,
                                  const float frequency,
                                  const float pitch);

    __host__ void releaseStructures();

public:
    __host__ CudaSaftZeroGraphNode();

    __host__ ~CudaSaftZeroGraphNode();

    __host__ void saftZero(const float *inputPtr,
                           float *outputPtr,
                           const cudaStream_t &stream,
                           const int apertureLen,
                           const int signalLen,
                           const float soundVelocity,
                           const float frequency,
                           const float pitch);
};
