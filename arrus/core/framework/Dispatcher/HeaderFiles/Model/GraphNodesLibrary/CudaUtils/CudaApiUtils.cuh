#ifndef __CUDA_API_UTILS__
#define __CUDA_API_UTILS__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

class CudaApiUtils {
public:

    template<class DataType, enum cudaTextureReadMode readMode>
    static __host__ bool
    bindTexture1DLinear(const texture <DataType, cudaTextureType1D, readMode> *texRef, const DataType *inputData,
                        const int dataCount, const cudaDeviceProp &deviceProp) {
        if(dataCount > deviceProp.maxTexture1DLinear)
            return false;

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<DataType>();
        CUDA_ASSERT(cudaBindTexture(NULL, texRef, inputData, &desc, dataCount * sizeof(DataType)));

        return true;
    }

    static __host__ cudaDeviceProp getCudaDeviceProps() {
        cudaDeviceProp props;
        int device;
        CUDA_ASSERT(cudaGetDevice(&device));
        CUDA_ASSERT(cudaGetDeviceProperties(&props, device));
        return props;
    }
};

#endif