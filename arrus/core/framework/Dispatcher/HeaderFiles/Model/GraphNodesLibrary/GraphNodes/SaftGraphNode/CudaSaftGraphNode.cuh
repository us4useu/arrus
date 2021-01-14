#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include <vector>

class CudaSaftGraphNode {
private:
    int inputApertureLen;
    int inputSignalLen;
    int allocatedApertureLen;
    int allocatedSignalLen;
    std::vector<float> allocatedAngles;

    float *devFkz;
    float *devForwardRealCompensation;
    float *devForwardImagCompensation;
    float *devInverseRealCompensation;
    float *devInverseImagCompensation;
    float *devObliquityFactor;
    float *devOutputBuffer;
    cufftHandle spatialPlan;
    cufftHandle temporalPlan;
    cufftComplex *devCufftData;
    cufftComplex *devCufftOutData;

    __host__ void allocStructures(const int apertureLen,
                                  const int signalLen,
                                  const std::vector<float> anglesInfo,
                                  const float t0,
                                  const float soundVelocity,
                                  const float frequency,
                                  const float pitch);

    __host__ void releaseStructures();

    std::vector<float> generateKx(const float pitch);

    std::vector<float> generateF0(const float fs);

    std::vector<float> generateFkz(const std::vector<float> f0,
                                   const std::vector<float> kx,
                                   const float soundVelocity,
                                   const float fs,
                                   const float pitch);

    std::vector<float> generateForwardCoefficients(const float soundVelocity,
                                                   const float fs,
                                                   const float pitch,
                                                   const float t0);

    std::vector<float> generateInverseCoefficients(const std::vector<float> kx,
                                                   const float soundVelocity,
                                                   const float fs,
                                                   const float pitch);

    std::vector<float> generateObliquityFactor(const std::vector<float> f0,
                                               const std::vector<float> fkz,
                                               const float fs);

public:
    __host__ CudaSaftGraphNode();

    __host__ ~CudaSaftGraphNode();

    __host__ void saft(const float *const inputPtr,
                       float *const outputPtr,
                       const int apertureLen,
                       const int signalLen,
                       const std::vector<float> anglesInfo,
                       const float t0,
                       const float soundVelocity,
                       const float frequency,
                       const float pitch,
                       const cudaStream_t &stream);
};
