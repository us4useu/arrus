#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

#define checkCudaErrors(val) check((val), __FILE__, __LINE__)

