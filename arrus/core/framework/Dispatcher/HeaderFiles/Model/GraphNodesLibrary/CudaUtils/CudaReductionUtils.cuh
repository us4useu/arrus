#ifndef __CUDA_REDUCTION_UTILS__
#define __CUDA_REDUCTION_UTILS__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include <algorithm>

template<typename T, typename BinaryFunction>
__global__ void gpuReduction(const T *inputData, const int dataCount, T *outProduct, const T initValue,
                             const BinaryFunction binaryFunction) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ T data[];

    T currentProduct = initValue;
    // First search over global memory
    for(int i = idx; i < dataCount; i += blockDim.x * gridDim.x)
        currentProduct = binaryFunction(inputData[i], currentProduct);

    data[threadIdx.x] = currentProduct;
    __syncthreads();

    // Next reduce over threads in block
    for(int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if(threadIdx.x < i) {
            T a = data[threadIdx.x];
            T b = data[threadIdx.x + i];
            data[threadIdx.x] = binaryFunction(a, b);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
        outProduct[blockIdx.x] = data[0];
}

template<typename T>
struct reductionMax {
    __device__ T operator()(const T &x, const T &y) const {
        return (x > y) ? x : y;
    }
};

template<typename T>
struct reductionMin {
    __device__ T operator()(const T &x, const T &y) const {
        return (x < y) ? x : y;
    }
};

class CudaReductionUtils {
private:
    int maxThreadsPerBlock;
    void *tempBuffer;
    int tempBufferSize;

    __host__ void allocMemory(const int tempBufferSize) {
        this->tempBufferSize = tempBufferSize;
        if(tempBuffer != nullptr)
            this->releaseMemory();
        CUDA_ASSERT(cudaMalloc((void **) &tempBuffer, this->tempBufferSize));
    }

    __host__ void releaseMemory() {
        if(tempBuffer != nullptr)
            CUDA_ASSERT(cudaFree(tempBuffer));
    }

    __host__ void getMaxBlockSize() {
        if(this->maxThreadsPerBlock == 0) {
            int device;
            CUDA_ASSERT(cudaGetDevice(&device));
            cudaDeviceProp props;
            CUDA_ASSERT(cudaGetDeviceProperties(&props, device));
            this->maxThreadsPerBlock = props.maxThreadsPerBlock;
        }
    }

public:
    __host__ CudaReductionUtils() {
        this->maxThreadsPerBlock = 0;
        this->tempBuffer = nullptr;
        this->tempBufferSize = 0;
    }

    __host__ ~CudaReductionUtils() {
        this->releaseMemory();
    }

    template<typename T, typename BinaryFunction>
    __host__ T *
    reduction(const T *inputData, const int dataCount, const T initValue, const BinaryFunction binaryFunction,
              const cudaStream_t &stream) {
        this->getMaxBlockSize();

        int currTempBufferSize = sizeof(T) * this->maxThreadsPerBlock;
        if(currTempBufferSize > this->tempBufferSize)
            this->allocMemory(currTempBufferSize);

        dim3 block(std::min(512, dataCount));
        dim3
        grid(std::min((int) ((dataCount + block.x - 1) / block.x), this->maxThreadsPerBlock));
        int externSharedMemorySize = block.x * sizeof(T);
        gpuReduction<T><<<grid, block, externSharedMemorySize, stream>>>(inputData, dataCount, (T *) tempBuffer,
                                                                         initValue, binaryFunction);
        CUDA_ASSERT(cudaGetLastError());

        if(grid.x == 1)
            return (T *) tempBuffer;

        block = dim3(this->maxThreadsPerBlock);
        externSharedMemorySize = block.x * sizeof(T);
        gpuReduction<T><<<dim3(1), block, externSharedMemorySize, stream>>>((T *) tempBuffer, grid.x, (T *) tempBuffer,
                                                                            initValue, binaryFunction);
        CUDA_ASSERT(cudaGetLastError());

        return (T *) tempBuffer;
    }
};

#endif
