#include "DigitalDownConversion.h"

namespace arrus_example_imaging {

// ------------------------------------------ Demodulation.
__global__ void gpuRfToIq(float2 *output, const float *input, const float sampleCoeff, const unsigned nSamples,
                          const unsigned totalNSamples) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= totalNSamples) {
        return;
    }
    float rfSample = input[idx];
    int sampleNumber = idx % nSamples;
    float cosinus, sinus;
    __sincosf(sampleCoeff * sampleNumber, &sinus, &cosinus);
    float2 iq;
    iq.x = 2.0f * rfSample * cosinus;
    iq.y = 2.0f * rfSample * sinus;
    output[idx] = iq;
}

void QuadratureDemodulationFunctor::operator()(NdArray &output, const NdArray &input, const unsigned nSamples,
                                               const unsigned totalNSamples, const float sampleCoeff,
                                               cudaStream_t stream) {
    dim3 filterBlockDim(512);
    dim3 filterGridDim((totalNSamples + filterBlockDim.x - 1) / filterBlockDim.x);
    gpuRfToIq<<<filterGridDim, filterBlockDim, 0, stream>>>(output.getPtr<float2>(), input.getConstPtr<float>(),
                                                            sampleCoeff, nSamples, totalNSamples);
    CUDA_ASSERT(cudaGetLastError());
}

// ------------------------------------------ Low-pass filter.
#define MAX_CIC_SIZE 512
__device__ __constant__ float gpuCicCoefficients[MAX_CIC_SIZE];
__global__ void gpuFirLp(float2 *__restrict__ output, const float2 *__restrict__ input, const unsigned nSamples,
                         const unsigned totalNSamples, const unsigned kernelWidth) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ch = idx / nSamples;
    int sample = idx % nSamples;

    extern __shared__ char sharedMemory[];

    float2 *cachedInputData = (float2 *) sharedMemory;
    // Cached input data stores all the input data which is convolved with given
    // filter.
    // That means, there should be enough input data from the last thread in
    // the thread group to compute convolution.
    // Thus the below condition localIdx < (blockDim.x + kernelWidth)
    // Cache input.
    for (int i = sample - kernelWidth / 2 - 1, localIdx = threadIdx.x; localIdx < (kernelWidth + blockDim.x);
         i += blockDim.x, localIdx += blockDim.x) {
        if (i < 0 || i >= nSamples) {
            cachedInputData[localIdx] = make_float2(0.0f, 0.0f);
        } else {
            cachedInputData[localIdx] = input[ch * nSamples + i];
        }
    }
    __syncthreads();
    if (idx >= totalNSamples) {
        return;
    }
    float2 result = make_float2(0.0f, 0.0f);

    int localN = threadIdx.x + kernelWidth;
    for (int i = 0; i < kernelWidth; ++i) {
        result.x += cachedInputData[localN - i].x * gpuCicCoefficients[i];
        result.y += cachedInputData[localN - i].y * gpuCicCoefficients[i];
    }
    output[idx] = result;
}

LowPassFilterFunctor::LowPassFilterFunctor(const NdArray &coeffs) {
    CUDA_ASSERT(cudaMemcpyToSymbol(gpuCicCoefficients, coeffs.getConstPtr<float>(), coeffs.getNBytes(), 0,
                                   cudaMemcpyHostToDevice));
}

void LowPassFilterFunctor::operator()(NdArray &output, const NdArray &input, unsigned nSamples, unsigned totalNSamples,
                                      unsigned nCoefficients, cudaStream_t stream) {
    dim3 filterBlockDim(512);
    dim3 filterGridDim((totalNSamples + filterBlockDim.x - 1) / filterBlockDim.x);
    unsigned sharedMemSize = (filterBlockDim.x + nCoefficients) * sizeof(float2);
    gpuFirLp<<<filterGridDim, filterBlockDim, sharedMemSize, stream>>>(
        output.getPtr<float2>(), input.getConstPtr<float2>(), nSamples, totalNSamples, nCoefficients);
    CUDA_ASSERT(cudaGetLastError());
}
// ------------------------------------------ Decimation.

__global__ void gpuDecimation(float2 *output, const float2 *input, const unsigned nSamples,
                              const unsigned totalNSamples, const unsigned decimationFactor) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= totalNSamples) {
        return;
    }
    // TODO the below could be computed on the kernel initialization
    int decimatedNSamples = (int) ceilf((float) nSamples / decimationFactor);
    int lineNr = idx / decimatedNSamples;  // output line number
    int sampleNr = idx % decimatedNSamples;// output sample number
    output[idx] = input[lineNr * nSamples + sampleNr * decimationFactor];
}

void DecimationFunctor::operator()(NdArray &output, const NdArray &input,
                                   const unsigned totalOutputNSamples, const unsigned nSamples,
                                   const unsigned decimationFactor,
                                   cudaStream_t stream) {
    dim3 block(512);
    dim3 grid((totalOutputNSamples + block.x - 1) / block.x);
    gpuDecimation<<<grid, block, 0, stream>>>(output.getPtr<float2>(), input.getConstPtr<float2>(), nSamples,
                                              totalOutputNSamples, decimationFactor);
    CUDA_ASSERT(cudaGetLastError());
}
}// namespace arrus_example_imaging
