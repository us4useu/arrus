#ifndef __CUDA_FILTER_RAW_DATA_GRAPH_NODE__
#define __CUDA_FILTER_RAW_DATA_GRAPH_NODE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

//extern __device__ __constant__ float deviceFeedforwardCoefficients[2048];
//extern __device__ __constant__ float deviceFeedbackCoefficients[2048];
//extern __device__ __constant__ float deviceFeedbackCoefficientsMatrix[2048];

#define IIR_DATA_BLOCK_COUNT 32

class CudaFilter1DGraphNode {
private:
    float *prologuesBuffer, *firOutputBuffer;
    float2 *prologuesIqBuffer, *firIqOutputBuffer;
    int prologuesBufferSize, firOutputBufferSize;
    int prologuesIqBufferSize, firIqOutputBufferSize;

    template<typename T>
    __host__ void releaseMemory(const T *ptr) {
        if(ptr != nullptr)
            CUDA_ASSERT(cudaFree((void *) ptr));
    }

    template<typename T>
    __host__ void allocMemory(const int bufferSize, T **ptr) {
        if(*ptr != nullptr)
            this->releaseMemory(*ptr);
        CUDA_ASSERT(cudaMalloc((void **) ptr, bufferSize));
    }

public:
    __host__ CudaFilter1DGraphNode();

    __host__ ~CudaFilter1DGraphNode();

    __host__ void
    fir(const float *inputPtr, float *outputPtr, const cudaStream_t &stream, const int feedforwardFilterSize,
        const int dataCount, const int batchLength);

    __host__ void
    iir(const float *inputPtr, float *outputPtr, const cudaStream_t &stream, const int feedforwardFilterSize,
        const int feedbackFilterSize, const int dataCount, const int batchLength);

    __host__ void
    firIq(const float2 *inputPtr, float2 *outputPtr, const cudaStream_t &stream, const int feedforwardFilterSize,
          const int dataCount, const int batchLength);

    __host__ void
    iirIq(const float2 *inputPtr, float2 *outputPtr, const cudaStream_t &stream, const int feedforwardFilterSize,
          const int feedbackFilterSize, const int dataCount, const int batchLength);
};

#endif
