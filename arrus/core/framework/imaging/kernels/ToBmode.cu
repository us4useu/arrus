#ifndef CPP_EXAMPLE_KERNELS_TOBMODE_CUH
#define CPP_EXAMPLE_KERNELS_TOBMODE_CUH

#include "ToBmode.h"

namespace arrus_example_imaging {

__global__ void gpuBMode(uint8_t *output, const float *input,
                         const float minDBLimit, const float maxDBLimit,
                         const int numberOfElements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numberOfElements) {
        return;
    }
    float pix = input[idx] + 1e-9;
    pix = 20.0f * log10f(pix);
    if (isnan(pix)) {
        pix = 0.0f;
    }
    // Cut on limits
    pix = fmaxf(minDBLimit, fminf(maxDBLimit, pix));
    // TODO Do a better remapping here.
    pix = pix-minDBLimit;
    pix = pix/(maxDBLimit-minDBLimit)*255;
    output[idx] = (uint8_t)pix;
}

void ToBModeFunctor::operator()(NdArray &output, const NdArray &input,
                                float minDbLimit, float maxDbLimit, cudaStream_t stream) {
    dim3 blockDim(512);
    unsigned numberOfElements = output.getNumberOfElements();
    dim3 gridDim((numberOfElements + blockDim.x - 1) / blockDim.x);
    gpuBMode<<<gridDim, blockDim, 0, stream >>>(
        output.getPtr<uint8_t>(), input.getConstPtr<float>(),
        minDbLimit, maxDbLimit, numberOfElements);
    CUDA_ASSERT(cudaGetLastError());
}

}

#endif //CPP_EXAMPLE_KERNELS_TOBMODE_CUH
