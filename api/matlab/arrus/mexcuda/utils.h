#ifndef ARRUS_UTILS_H
#define ARRUS_UTILS_H

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

#endif //ARRUS_UTILS_H
