#ifndef CPP_EXAMPLE_KERNELS_TRANSPOSE_H
#define CPP_EXAMPLE_KERNELS_TRANSPOSE_H

#include "Transpose.h"

namespace arrus_example_imaging {

#define GPU_TRANSPOSE_TILE_DIM 32

__global__ void gpuTranspose(short *out, const short *in, const unsigned nColumns, const unsigned nRows) {
    __shared__ short tile[GPU_TRANSPOSE_TILE_DIM][GPU_TRANSPOSE_TILE_DIM + 1];

    unsigned xIndex = blockIdx.x * GPU_TRANSPOSE_TILE_DIM + threadIdx.x;
    unsigned yIndex = blockIdx.y * GPU_TRANSPOSE_TILE_DIM + threadIdx.y;
    unsigned zIndex = blockIdx.z;
    unsigned index_in = xIndex + yIndex * nColumns + zIndex * nColumns * nRows;

    if ((xIndex < nColumns) && (yIndex < nRows)) {
        tile[threadIdx.y][threadIdx.x] = in[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * GPU_TRANSPOSE_TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * GPU_TRANSPOSE_TILE_DIM + threadIdx.y;
    unsigned index_out = xIndex + yIndex * nRows + zIndex * nColumns * nRows;

    if ((xIndex < nRows) && (yIndex < nColumns)) {
        out[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

void TransposeFunctor::operator()(NdArray &output, const NdArray &input,
                                  unsigned int nMatrices, unsigned int nRows, unsigned int nColumns,
                                  cudaStream_t stream) {
    dim3 block(GPU_TRANSPOSE_TILE_DIM, GPU_TRANSPOSE_TILE_DIM);
    dim3 grid((nColumns - 1) / block.x + 1, (nRows - 1) / block.y + 1, nMatrices);
    gpuTranspose<<<grid, block, 0, stream>>>(output.getPtr<short>(), input.getConstPtr<short>(), nColumns, nRows);
    CUDA_ASSERT(cudaGetLastError());
}
}

#endif //CPP_EXAMPLE_KERNELS_TRANSPOSE_H