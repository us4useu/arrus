#ifndef CPP_EXAMPLE_CUDAUTILS_H
#define CPP_EXAMPLE_CUDAUTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>

#define CUDA_ASSERT(opr) { \
    cudaError_t ret = (opr); \
    if (cudaSuccess != ret) { \
        fprintf(stderr, "CUDA Error: %d %s %s\n", __LINE__, __FILE__, cudaGetErrorString(ret)); \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(ret)) + ", code: " + std::to_string(ret)); \
    } \
}

#define CUDA_ASSERT_NO_THROW(opr) { \
    cudaError_t ret = (opr); \
    if (cudaSuccess != ret) { \
        fprintf(stderr, "CUDA Error: %d %s %s\n", __LINE__, __FILE__, cudaGetErrorString(ret)); \
    } \
}

#endif //CPP_EXAMPLE_CUDAUTILS_H
