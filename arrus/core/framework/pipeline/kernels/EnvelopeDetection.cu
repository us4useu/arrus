#ifndef CPP_EXAMPLE_KERNELS_ENVELOPEDETECTION_CUH
#define CPP_EXAMPLE_KERNELS_ENVELOPEDETECTION_CUH

#include "EnvelopeDetection.h"

namespace arrus_example_imaging {
__global__ void gpuEnvelopeDetection(float *output, const float2 *input, const unsigned totalNSamples) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= totalNSamples) {
        return;
    }
    float2 value = input[idx];
    output[idx] = hypotf(value.x, value.y);
}

void EnvelopeDetectionFunctor::operator()(NdArray &output, const NdArray &input, cudaStream_t stream) {
    dim3 block(512);
    unsigned totalNSamples = input.getNumberOfElements();
    dim3 grid((totalNSamples + block.x - 1) / block.x);
    gpuEnvelopeDetection<<<grid, block, 0, stream>>>(output.getPtr<float>(), input.getConstPtr<float2>(),
                                                     totalNSamples);
    CUDA_ASSERT(cudaGetLastError());
}
}// namespace arrus_example_imaging

#endif
