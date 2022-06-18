#include "BandpassFilter.h"

namespace arrus_example_imaging {

#define MAX_FIR_SIZE 512

__device__ __constant__ float gpuFirCoefficients[MAX_FIR_SIZE];

__global__ void gpuFir(float *__restrict__ output, const short *__restrict__ input, const unsigned nSamples,
                       const unsigned totalNSamples, const unsigned kernelWidth) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ch = idx / nSamples;
    int sample = idx % nSamples;

    extern __shared__ char sharedMemory[];

    auto *cachedInputData = (short *) sharedMemory;
    // Cached input data stores all the input data which is convolved with given
    // filter.
    // That means, there should be enough input data from the last thread in
    // the thread group to compute convolution.
    // Thus the below condition localIdx < (blockDim.x + kernelWidth)
    // Cache input.
    for (int i = sample - kernelWidth / 2 - 1, localIdx = threadIdx.x; localIdx < (kernelWidth + blockDim.x);
         i += blockDim.x, localIdx += blockDim.x) {
        if (i < 0 || i >= nSamples) {
            cachedInputData[localIdx] = 0;
        } else {
            cachedInputData[localIdx] = input[ch * nSamples + i];
        }
    }

    __syncthreads();
    if (idx >= totalNSamples) {
        return;
    }
    float result = 0.0f;

    int localN = threadIdx.x + kernelWidth;
    for (int i = 0; i < kernelWidth; ++i) {
        result += cachedInputData[localN - i] * gpuFirCoefficients[i];
    }

    output[idx] = result;
}

BandpassFilterFunctor::BandpassFilterFunctor(const NdArray &coeffs) {
    CUDA_ASSERT(cudaMemcpyToSymbol(
        gpuFirCoefficients,
        coeffs.getConstPtr<void>(),
        coeffs.getNBytes(),
        0, cudaMemcpyHostToDevice));
}


void BandpassFilterFunctor::operator()(NdArray &output, const NdArray &input,
                                       unsigned totalNSamples, unsigned nCoefficients, unsigned nSamples,
                                       cudaStream_t stream) {
    dim3 filterBlockDim(512);
    dim3 filterGridDim((totalNSamples + filterBlockDim.x - 1) / filterBlockDim.x);
    unsigned sharedMemorySize = (filterBlockDim.x + nCoefficients) * sizeof(short);
    gpuFir<<<filterGridDim, filterBlockDim, sharedMemorySize, stream>>>(
        output.getPtr<float>(), input.getConstPtr<short>(), nSamples, totalNSamples, nCoefficients);
    CUDA_ASSERT(cudaGetLastError());
}

/**
 * FIR filter. NOTE: there should be only one instance of this kernel in a single
 * imaging pipeline.
 * The constraint on the number of instances is due to the usage of
 * global constant memory to store filter coefficients.
 */
//class FirFilterSingleton : public Kernel {
//public:
//
//    KernelInitResult prepare(const KernelInitContext &ctx) override {
//        auto &inputShape = ctx.getInputShape();
//        if (ctx.getInputShape().size() != 3) {
//            throw std::runtime_error(
//                "Currently fir filter works only with 3D arrays");
//        }
//        this->totalNSamples = inputShape[0] * inputShape[1] * inputShape[2];
//        this->nSamples = inputShape[2];
//        return KernelInitResult(
//            inputShape, NdArray::DataType::FLOAT32,
//            ctx.getInputSamplingFrequency());
//    }
//
//
//    explicit FirFilterSingleton(const std::vector<float> &coefficients) {
//        this->nCoefficients = coefficients.size();
//        if (coefficients.size() > MAX_FIR_SIZE) {
//            throw std::runtime_error("Exceeded maximum number of "
//                                     "filter coefficients");
//        }
//        CUDA_ASSERT(cudaMemcpyToSymbol(
//            gpuFirCoefficients,
//            coefficients.data(),
//            coefficients.size() * sizeof(float),
//            0, cudaMemcpyHostToDevice));
//
//    }
//
//private:
//    unsigned totalNSamples{0};
//    unsigned nSamples{0};
//    unsigned nCoefficients{0};
//};
}// namespace arrus_example_imaging