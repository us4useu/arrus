#ifndef __CUDA_ASSERT_CUH__
#define __CUDA_ASSERT_CUH__

#include "Utils/DispatcherLogger.h"
#include <cstdlib>
#include <cstdio>
#include <cufft.h>
#include <nvrtc.h>

#define CUDA_ASSERT(opr) { \
    cudaError_t ret = (opr); \
    if (cudaSuccess != ret) { \
        fprintf(stderr, "Error: %d %s %s\n", __LINE__, __FILE__, cudaGetErrorString(ret)); \
        DISPATCHER_LOG(DispatcherLogType::FATAL, std::string("Error: ") + std::to_string(__LINE__) + " " + std::string(__FILE__) + " " + cudaGetErrorString(ret)); \
    } \
}

#define CUFFT_ASSERT(opr) { \
    cufftResult ret = (opr); \
    if (cudaSuccess != ret) { \
        fprintf(stderr, "CuFFT Error: %d %s %d\n", __LINE__, __FILE__, ret); \
        DISPATCHER_LOG(DispatcherLogType::FATAL, std::string("CuFFT Error: ") + std::to_string(__LINE__) + " " + std::string(__FILE__) + " " + std::to_string(ret)); \
    } \
}

#define CU_ASSERT(opr) { \
    CUresult ret = opr; \
    if (ret != CUDA_SUCCESS) { \
        const char* msg; \
        cuGetErrorName(ret, &msg); \
        fprintf(stderr, "Error: %d %s %s\n", __LINE__, __FILE__, msg); \
        DISPATCHER_LOG(DispatcherLogType::FATAL, std::string("Error: ") + std::to_string(__LINE__) + " " + std::string(__FILE__) + " " + msg); \
    } \
}

#define NVRTC_ASSERT(opr) { \
    nvrtcResult ret = opr; \
    if (ret != NVRTC_SUCCESS) { \
        fprintf(stderr, "Error: %d %s %s\n", __LINE__, __FILE__, nvrtcGetErrorString(ret)); \
        DISPATCHER_LOG(DispatcherLogType::FATAL, std::string("Error: ") + std::to_string(__LINE__) + " " + std::string(__FILE__) + " " + nvrtcGetErrorString(ret)); \
    } \
}

#endif /* __CUDA_ASSERT_CUH__ */