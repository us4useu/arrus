#ifndef __CUDA_TRANSPOSITION_UTILS__
#define __CUDA_TRANSPOSITION_UTILS__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

#define TRANSPOSITION_TILE_DIM 32

template<typename T, typename K>
__global__ void gpuTranspose(const T *in, K *out, const int width, const int height) {
    __shared__ K tile[TRANSPOSITION_TILE_DIM][TRANSPOSITION_TILE_DIM + 1];

    int xIndex = blockIdx.x * TRANSPOSITION_TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TRANSPOSITION_TILE_DIM + threadIdx.y;
    int zIndex = blockIdx.z;
    int index_in = xIndex + yIndex * width + zIndex * width * height;

    if((xIndex < width) && (yIndex < height))
        tile[threadIdx.y][threadIdx.x] = in[index_in];

    __syncthreads();

    xIndex = blockIdx.y * TRANSPOSITION_TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TRANSPOSITION_TILE_DIM + threadIdx.y;
    int index_out = xIndex + yIndex * height + zIndex * width * height;

    if((xIndex < height) && (yIndex < width))
        out[index_out] = tile[threadIdx.x][threadIdx.y];
}

class CudaTranspositionUtils {
public:
    template<typename T, typename K>
    __host__ static void transpose(const T *in, K *out, const int width, const int height, const cudaStream_t &stream,
                                   const int numberOfTranspositions = 1) {
        dim3 block(TRANSPOSITION_TILE_DIM, TRANSPOSITION_TILE_DIM);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, numberOfTranspositions);
        gpuTranspose << < grid, block, 0, stream >> >(in, out, width, height);
        CUDA_ASSERT(cudaGetLastError());
    }
};

#endif