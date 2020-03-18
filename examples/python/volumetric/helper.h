#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

// Based on CUDA examples common code.
template <typename T>
void check(T result, const char *const file, const int line) {
    if(result) {
        fprintf(stderr, "CUDA error at %s:%d, error %s (code %d): %s\n",
                file, line, cudaGetErrorName(result), result, 
                cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

template <>
void check(cufftResult result, const char *const file, const int line) {
    if(result != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error at %s:%d, error code %d\n",
                file, line, result);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), __FILE__, __LINE__)

