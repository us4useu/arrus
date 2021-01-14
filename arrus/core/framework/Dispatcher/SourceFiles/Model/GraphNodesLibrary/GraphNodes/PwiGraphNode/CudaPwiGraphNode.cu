#include "Model/GraphNodesLibrary/GraphNodes/PwiGraphNode/CudaPwiGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaCommonsUtils.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaApiUtils.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaVectorMathUtils.cuh"
#include <math_constants.h>

texture<float, cudaTextureType1D, cudaReadModeElementType> texPwiRfInputData;
texture <float2, cudaTextureType1D, cudaReadModeElementType> texPwiIqInputData;

__forceinline__ __device__ float
getHanningWeight(const float *hanningWindow, int &hanningWindowIdx, const int anglesCount) {
    hanningWindowIdx = hanningWindowIdx + 1;
    return hanningWindow[hanningWindowIdx / anglesCount];
}

template<typename T, bool useTexture>
__forceinline__ __device__ T
getInterpVal(const T *input, const float realSample, const float sampleOffset, const float totalDistance,
             const float startDepth,
             const float soundVelocity, const float transmitFrequency) {
    return CudaCommonsUtils::getInterpolated1DValue(input, realSample, sampleOffset);
}

template<>
__forceinline__ __device__ float2
getInterpVal<float2, true>(const float2 *input, const float realSample, const float sampleOffset,
                           const float totalDistance,
                           const float startDepth, const float soundVelocity, const float transmitFrequency) {
    float2 currIqSample = CudaCommonsUtils::getInterpolated1DValue(texPwiIqInputData, realSample, sampleOffset);
    return CudaCommonsUtils::getCurrentIQSampleValue(currIqSample, transmitFrequency, totalDistance, startDepth,
                                                     soundVelocity);
}

template<>
__forceinline__ __device__ float
getInterpVal<float, true>(const float *input, const float realSample, const float sampleOffset,
                          const float totalDistance,
                          const float startDepth, const float soundVelocity, const float transmitFrequency) {
    return CudaCommonsUtils::getInterpolated1DValue(texPwiRfInputData, realSample, sampleOffset);
}

template<>
__forceinline__ __device__ float2
getInterpVal<float2, false>(const float2 *input, const float realSample, const float sampleOffset,
                            const float totalDistance,
                            const float startDepth, const float soundVelocity, const float transmitFrequency) {
    float2 currIqSample = CudaCommonsUtils::getInterpolated1DValue(input, realSample, sampleOffset);
    return CudaCommonsUtils::getCurrentIQSampleValue(currIqSample, transmitFrequency, totalDistance, startDepth,
                                                     soundVelocity);
}

template<PWI_APODIZATION apod, typename T, bool useTexture, typename OUT_T>
__global__ void gpuPwi(const T *input, OUT_T *output, const int resultWidthInPixels, const int resultHeightInPixels,
                       const int anglesCount, const float soundVelocity,
                       const float receiverWidth, const int receiversCount, const float samplingFrequency,
                       const int samplesCount,
                       const float startDepth, const float *anglesInfo, const float *hanningWindow,
                       const CudaPwiPixelMap pixelMap, const float transmitFrequency) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float realX = (float) x / (float) (resultWidthInPixels - 1) * pixelMap.width + pixelMap.x;
    float realY = (float) y / (float) (resultHeightInPixels - 1) * pixelMap.height + pixelMap.y;
    float realX1 = realX - receiverWidth;

    float startAngleSinus = sinf(fminf(fmaxf(CUDART_PIO2 - atan2(realY, realX), -CUDART_PIO2), CUDART_PIO2));
    float endAngleSinus = sinf(fminf(fmaxf(-CUDART_PIO2 + atan2(realY, -realX1), -CUDART_PIO2), CUDART_PIO2));

    T result = makeZeroValue<T>();

    extern __shared__ float cachedSinCos[];

    int localIdx = threadIdx.x + threadIdx.y * blockDim.x;
    for(int i = localIdx; i < anglesCount * 2; i += blockDim.x * blockDim.y)
        cachedSinCos[i] = anglesInfo[i];

    __syncthreads();

    int startReceiver = 0, endReceiver = receiversCount, hanningWindowIdx;
    if(apod == PWI_APODIZATION::HANN) {
        CudaCommonsUtils::getHanningReceiversRange(realX, receiverWidth, receiversCount, startReceiver, endReceiver,
                                                   hanningWindowIdx);
        --hanningWindowIdx;
    }

    int anglesCompensation = anglesCount;
    for(int r = startReceiver; r < endReceiver; ++r) {
        float realReceiver = CudaCommonsUtils::getRealReceiver(r, receiversCount, receiverWidth);
        float receiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, realReceiver);
        float tasinkevychConstant = CudaCommonsUtils::getTasinkevychConstant(receiverWidth, receiversCount,
                                                                             soundVelocity, transmitFrequency);
        anglesCompensation = 0;
        for(int i = 0; i < anglesCount; ++i) {
            float sinus = cachedSinCos[(i << 1)];
            float cosinus = cachedSinCos[(i << 1) + 1];
            // cut corners outside plain wave front
            if((startAngleSinus >= sinus) && (endAngleSinus <= sinus)) {
                float newX = (sinus < 0.0f) ? realX1 : realX;
                float transmitDistance = fma(realY, cosinus, newX * sinus);
                float totalDistance = transmitDistance + receiveDistance;
                float realSample = CudaCommonsUtils::getRealSample(totalDistance - startDepth * 2.0f, soundVelocity,
                                                                   samplingFrequency);

                T currResult = makeZeroValue<T>();
                if(((int) realSample < samplesCount) && ((int) realSample >= 0)) {
                    float sampleOffset = samplesCount * r + receiversCount * samplesCount * i;
                    currResult = getInterpVal<T, useTexture>(input, realSample, sampleOffset, totalDistance, startDepth,
                                                             soundVelocity, transmitFrequency);
                }

                if(apod == PWI_APODIZATION::HANN)
                    currResult *= getHanningWeight(hanningWindow, hanningWindowIdx, anglesCount);
                else if(apod == PWI_APODIZATION::TAS)
                    currResult *= fabsf(
                        CudaCommonsUtils::getTasinkevychWeight(realX, realY, realReceiver, tasinkevychConstant,
                                                               receiveDistance));

                result += currResult;
                ++anglesCompensation;
            }
        }
    }

    //output[x + y * blockDim.x * gridDim.x] = CudaCommonsUtils::getComplexModulus(result) * anglesCount / fmaxf(anglesCompensation, 1.0f);
    output[x + y * blockDim.x * gridDim.x] = result * anglesCount / fmaxf(anglesCompensation, 1.0f);

}

void CudaPwiGraphNode::pwiRf(const float *inputData, float *output, const cudaStream_t &stream,
                             const int resultWidthInPixels, const int resultHeightInPixels, const int anglesCount,
                             const float soundVelocity, const float receiverWidth, const int receiversCount,
                             const float samplingFrequency, const int samplesCount,
                             const float startDepth, const float *anglesInfo, const float *hanningWindow,
                             const CudaPwiPixelMap pixelMap, const float transmitFrequency,
                             const PWI_APODIZATION apod) {
    this->loadCudaDeviceProps();
    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texPwiRfInputData, inputData,
                                                             samplesCount * receiversCount * anglesCount,
                                                             this->deviceProp.get());

    dim3 blockDim(8, 32);
    dim3 gridDim(resultWidthInPixels / blockDim.x, resultHeightInPixels / blockDim.y);
    int externalSharedMemory = sizeof(float) * anglesCount * 2;
    if(isTextureBinded) {
        if(apod == PWI_APODIZATION::NONE)
            gpuPwi<PWI_APODIZATION::NONE, float, true, float> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        else if(apod == PWI_APODIZATION::HANN)
            gpuPwi<PWI_APODIZATION::HANN, float, true, float> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        else if(apod == PWI_APODIZATION::TAS)
            gpuPwi<PWI_APODIZATION::TAS, float, true, float> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        CUDA_ASSERT(cudaUnbindTexture(&texPwiRfInputData));
    } else {
        if(apod == PWI_APODIZATION::NONE)
            gpuPwi<PWI_APODIZATION::NONE, float, false, float> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        else if(apod == PWI_APODIZATION::HANN)
            gpuPwi<PWI_APODIZATION::HANN, float, false, float> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        else if(apod == PWI_APODIZATION::TAS)
            gpuPwi<PWI_APODIZATION::TAS, float, false, float> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
    }
    CUDA_ASSERT(cudaGetLastError());
}

void CudaPwiGraphNode::pwiIq(const float2 *inputData, float2 *output, const cudaStream_t &stream,
                             const int resultWidthInPixels, const int resultHeightInPixels, const int anglesCount,
                             const float soundVelocity, const float receiverWidth, const int receiversCount,
                             const float samplingFrequency, const int samplesCount,
                             const float startDepth, const float *anglesInfo, const float *hanningWindow,
                             const CudaPwiPixelMap pixelMap, const float transmitFrequency,
                             const PWI_APODIZATION apod) {
    this->loadCudaDeviceProps();
    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texPwiIqInputData, inputData,
                                                             samplesCount * receiversCount * anglesCount,
                                                             this->deviceProp.get());

    dim3 blockDim(8, 32);
    dim3 gridDim(resultWidthInPixels / blockDim.x, resultHeightInPixels / blockDim.y);
    int externalSharedMemory = sizeof(float) * anglesCount * 2;
    if(isTextureBinded) {
        if(apod == PWI_APODIZATION::NONE)
            gpuPwi<PWI_APODIZATION::NONE, float2, true, float2> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        else if(apod == PWI_APODIZATION::HANN)
            gpuPwi<PWI_APODIZATION::HANN, float2, true, float2> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        else if(apod == PWI_APODIZATION::TAS)
            gpuPwi<PWI_APODIZATION::TAS, float2, true, float2> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        CUDA_ASSERT(cudaUnbindTexture(&texPwiIqInputData));
    } else {
        if(apod == PWI_APODIZATION::NONE)
            gpuPwi<PWI_APODIZATION::NONE, float2, false, float2> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        else if(apod == PWI_APODIZATION::HANN)
            gpuPwi<PWI_APODIZATION::HANN, float2, false, float2> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
        else if(apod == PWI_APODIZATION::TAS)
            gpuPwi<PWI_APODIZATION::TAS, float2, false, float2> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputData, output, resultWidthInPixels, resultHeightInPixels, anglesCount, soundVelocity,
            receiverWidth, receiversCount, samplingFrequency, samplesCount, startDepth, anglesInfo,
            hanningWindow, pixelMap, transmitFrequency);
    }
    CUDA_ASSERT(cudaGetLastError());
}

void CudaPwiGraphNode::loadCudaDeviceProps() {
    if(!this->deviceProp) {
        this->deviceProp = boost::optional<cudaDeviceProp>(CudaApiUtils::getCudaDeviceProps());
    }
}