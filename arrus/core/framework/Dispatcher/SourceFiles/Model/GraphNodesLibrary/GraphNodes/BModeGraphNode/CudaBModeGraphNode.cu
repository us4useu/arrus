#include "Model/GraphNodesLibrary/GraphNodes/BModeGraphNode/CudaBModeGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaCommonsUtils.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaReductionUtils.cuh"

static CudaReductionUtils minCudaReductionUtils, maxCudaReductionUtils;

__global__ void
gpuBMode(const float *input, float *output, const float *minValue, const float *maxValue, const float minDBLimit,
         const float maxDBLimit, const int maxThreads,
         const float userMaxValue) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= maxThreads)
        return;

    // Normalize to (0, 1+epsilon]
    float currMaxValue = (userMaxValue == FLT_MAX) ? maxValue[0] : userMaxValue;
    float pix = (input[idx] - minValue[0]) / (currMaxValue - minValue[0]) + 1e-9;
    pix = 20.0f * log10f(pix);

    // Cut on limits
    pix = fmaxf(minDBLimit, fminf(maxDBLimit, pix));

    float resultPix = (pix - minDBLimit) / (maxDBLimit - minDBLimit);

    if(isnan(resultPix))
        resultPix = 1.0f;

    output[idx] = resultPix;
}

__global__ void gpuComplexModulus(const float2 *input, float *output, const int maxThreads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= maxThreads)
        return;

    output[idx] = CudaCommonsUtils::getComplexModulus(input[idx]);
}

void CudaBModeGraphNode::convertToBMode(const float *inputPtr, float *outputPtr, const cudaStream_t &stream,
                                        const float minDBLimit, const float maxDBLimit, const int dataCount,
                                        const float maxDataValue) {
    float *maxValue = (maxDataValue == FLT_MAX) ? maxCudaReductionUtils.reduction(inputPtr, dataCount, (float) INT_MIN,
                                                                                  reductionMax<float>(), stream)
                                                : nullptr;
    float *minValue = minCudaReductionUtils.reduction(inputPtr, dataCount, (float) INT_MAX, reductionMin<float>(),
                                                      stream);

    dim3 block(512);
    dim3 grid((dataCount + block.x - 1) / block.x);
    gpuBMode << <
    grid, block, 0, stream >> >(inputPtr, outputPtr, minValue, maxValue, minDBLimit, maxDBLimit, dataCount, maxDataValue);
    CUDA_ASSERT(cudaGetLastError());
}

void CudaBModeGraphNode::convertToBModeIq(const float2 *inputPtr, float *complexModulusPtr, float *outputPtr,
                                          const cudaStream_t &stream, const float minDBLimit, const float maxDBLimit,
                                          const int dataCount,
                                          const float maxDataValue) {
    dim3 block(512);
    dim3 grid((dataCount + block.x - 1) / block.x);
    gpuComplexModulus << < grid, block, 0, stream >> >(inputPtr, complexModulusPtr, dataCount);
    CUDA_ASSERT(cudaGetLastError());

    float *maxValue = (maxDataValue == FLT_MAX) ? maxCudaReductionUtils.reduction(complexModulusPtr, dataCount,
                                                                                  (float) INT_MIN,
                                                                                  reductionMax<float>(), stream)
                                                : nullptr;
    float *minValue = minCudaReductionUtils.reduction(complexModulusPtr, dataCount, (float) INT_MAX,
                                                      reductionMin<float>(), stream);

    gpuBMode << <
    grid, block, 0, stream >> >(complexModulusPtr, outputPtr, minValue, maxValue, minDBLimit, maxDBLimit, dataCount, maxDataValue);
    CUDA_ASSERT(cudaGetLastError());
}
