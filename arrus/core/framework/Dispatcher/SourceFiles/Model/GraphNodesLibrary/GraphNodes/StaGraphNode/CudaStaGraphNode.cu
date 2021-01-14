#include "Model/GraphNodesLibrary/GraphNodes/StaGraphNode/CudaStaGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaCommonsUtils.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaApiUtils.cuh"
#include <algorithm>

texture<float, cudaTextureType1D, cudaReadModeElementType> texStaRfInputData;
texture <float2, cudaTextureType1D, cudaReadModeElementType> texStaIqInputData;

#define OPENING_ANGLE_TANGENS 0.577350269f // 30 degrees

__forceinline__ __device__ float getHanningWeight(const float *hanningWindow, int &hanningWindowIdx) {
    hanningWindowIdx = hanningWindowIdx + 1;
    return hanningWindow[hanningWindowIdx];
}

template<bool useTexture, typename T>
__forceinline__ __device__ T
getInterpVal(const T *input, const float realSample, const float sampleOffset, const float totalDistance,
             const float startDepth,
             const float soundVelocity, const float transmitFrequency) {
    return CudaCommonsUtils::getInterpolated1DValue(input, realSample, sampleOffset);
}

template<>
__forceinline__ __device__ float2
getInterpVal<true, float2>(const float2 *input, const float realSample, const float sampleOffset,
                           const float totalDistance,
                           const float startDepth, const float soundVelocity, const float transmitFrequency) {
    float2 currIqSample = CudaCommonsUtils::getInterpolated1DValue(texStaIqInputData, realSample, sampleOffset);
    return CudaCommonsUtils::getCurrentIQSampleValue(currIqSample, transmitFrequency, totalDistance, startDepth,
                                                     soundVelocity);
}

template<>
__forceinline__ __device__ float
getInterpVal<true, float>(const float *input, const float realSample, const float sampleOffset,
                          const float totalDistance,
                          const float startDepth, const float soundVelocity, const float transmitFrequency) {
    return CudaCommonsUtils::getInterpolated1DValue(texStaRfInputData, realSample, sampleOffset);
}

template<>
__forceinline__ __device__ float2
getInterpVal<false, float2>(const float2 *input, const float realSample, const float sampleOffset,
                            const float totalDistance,
                            const float startDepth, const float soundVelocity, const float transmitFrequency) {
    float2 currIqSample = CudaCommonsUtils::getInterpolated1DValue(input, realSample, sampleOffset);
    return CudaCommonsUtils::getCurrentIQSampleValue(currIqSample, transmitFrequency, totalDistance, startDepth,
                                                     soundVelocity);
}

template<STA_APODIZATION apod, bool useTexture, typename T>
__forceinline__ __device__ T
summSamplesOverReceivers(const T *input, const int startReceiver, const int endReceiver, const float realX,
                         const float realY, const int receiversCount,
                         const float receiverWidth, const int inputOffset, const int samplesCount,
                         const float soundVelocity, const float samplingFrequency,
                         const float startDepth, const float transmitDistance, int hanningWindowIdx,
                         const float *hanningWindow,
                         const float tasinkevychConstant, const float transmitTasinkevychWeight,
                         const float transmitFrequency) {
    T result = makeZeroValue<T>();
    for(int r = startReceiver; r < endReceiver; ++r) {
        float realReceiver = CudaCommonsUtils::getRealReceiver(r, receiversCount, receiverWidth);
        float receiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, realReceiver);
        float totalDistance = transmitDistance + receiveDistance;
        float realSample = CudaCommonsUtils::getRealSample(totalDistance - startDepth * 2.0f, soundVelocity,
                                                           samplingFrequency);

        T currResult = makeZeroValue<T>();
        if(((int) realSample < samplesCount) && ((int) realSample >= 0))
            currResult = getInterpVal<useTexture>(input, realSample, inputOffset + samplesCount * r, totalDistance,
                                                  startDepth, soundVelocity, transmitFrequency);

        if(apod == STA_APODIZATION::HANN)
            currResult *= getHanningWeight(hanningWindow, hanningWindowIdx);
        else if(apod == STA_APODIZATION::TAS)
            currResult *= fabsf(transmitTasinkevychWeight *
                                CudaCommonsUtils::getTasinkevychWeight(realX, realY, realReceiver, tasinkevychConstant,
                                                                       receiveDistance));
        result += currResult;
    }
    return result;
}

template<STA_APODIZATION apod, bool useTexture, typename T>
struct OneTransmitSta {
    __forceinline__ __device__ T
    operator()(const T *input, const int receiversCount, const int samplesCount, const float receiverWidth,
               const float soundVelocity, const float samplingFrequency,
               const float startDepth, const float *hanningWindow, const int inputOffset, const float transmitDistance,
               const float realX, const float realY,
               const float realReceiver, const float transmitFrequency) {
        return summSamplesOverReceivers<apod, useTexture>(input, 0, receiversCount, realX, realY, receiversCount,
                                                          receiverWidth, inputOffset, samplesCount, soundVelocity,
                                                          samplingFrequency,
                                                          startDepth, transmitDistance, 0, hanningWindow, 0.0f, 1.0f,
                                                          transmitFrequency);
    }
};

template<bool useTexture, typename T>
struct OneTransmitSta<STA_APODIZATION::TAS, useTexture, T> {
    __forceinline__ __device__ T
    operator()(const T *input, const int receiversCount, const int samplesCount, const float receiverWidth,
               const float soundVelocity, const float samplingFrequency,
               const float startDepth, const float *hanningWindow, const int inputOffset, const float transmitDistance,
               const float realX, const float realY,
               const float realReceiver, const float transmitFrequency) {
        float tasinkevychConstant = CudaCommonsUtils::getTasinkevychConstant(receiverWidth, receiversCount,
                                                                             soundVelocity, transmitFrequency);
        float transmitTasinkevychWeight = CudaCommonsUtils::getTasinkevychWeight(realX, realY, realReceiver,
                                                                                 tasinkevychConstant, transmitDistance);
        return summSamplesOverReceivers<STA_APODIZATION::TAS, useTexture>(input, 0, receiversCount, realX, realY,
                                                                          receiversCount, receiverWidth, inputOffset,
                                                                          samplesCount, soundVelocity,
                                                                          samplingFrequency,
                                                                          startDepth, transmitDistance, 0,
                                                                          hanningWindow, tasinkevychConstant,
                                                                          transmitTasinkevychWeight, transmitFrequency);
    }
};

template<bool useTexture, typename T>
struct OneTransmitSta<STA_APODIZATION::HANN, useTexture, T> {
    __forceinline__ __device__ T
    operator()(const T *input, const int receiversCount, const int samplesCount, const float receiverWidth,
               const float soundVelocity, const float samplingFrequency,
               const float startDepth, const float *hanningWindow, const int inputOffset, const float transmitDistance,
               const float realX, const float realY,
               const float realReceiver, const float transmitFrequency) {
        float transmitOpeningAngleTangens = fabsf((realReceiver - realX) / realY);
        T result = makeZeroValue<T>();
        if(transmitOpeningAngleTangens <= OPENING_ANGLE_TANGENS) {
            int startReceiver, endReceiver, hanningWindowIdx;
            CudaCommonsUtils::getHanningReceiversRange(realX, receiverWidth, receiversCount, startReceiver, endReceiver,
                                                       hanningWindowIdx);
            CudaCommonsUtils::limitReceiversByOpeningAngle(realX, realY, receiverWidth, receiversCount, startReceiver,
                                                           endReceiver, hanningWindowIdx, OPENING_ANGLE_TANGENS);

            result = summSamplesOverReceivers<STA_APODIZATION::HANN, useTexture>(input, startReceiver, endReceiver,
                                                                                 realX, realY, receiversCount,
                                                                                 receiverWidth, inputOffset,
                                                                                 samplesCount, soundVelocity,
                                                                                 samplingFrequency,
                                                                                 startDepth, transmitDistance,
                                                                                 hanningWindowIdx - 1, hanningWindow,
                                                                                 0.0f, 1.0f, transmitFrequency);
        }
        return result;
    }
};

template<STA_APODIZATION apod, bool useTexture, typename T>
__global__ void
gpuSta(const T *input, float *output, const int widthInPixels, const int heightInPixels, const int receiversCount,
       const int samplesCount, const float receiverWidth, const float soundVelocity, const float samplingFrequency,
       const float startDepth,
       const float *transmittersIndexes, const int transmittersCount, const float *hanningWindow,
       const float transmitFrequency,
       const CudaStaPixelMap pixelMap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float realX = (float) x / (float) (widthInPixels - 1) * pixelMap.width + pixelMap.x;
    float realY = (float) y / (float) (heightInPixels - 1) * pixelMap.height + pixelMap.y;

    T result = makeZeroValue<T>();

    extern __shared__ float cachedTransmittersIndexes[];

    int localIdx = threadIdx.x + threadIdx.y * blockDim.x;
    for(int i = localIdx; i < transmittersCount; i += blockDim.x * blockDim.y)
        cachedTransmittersIndexes[i] = transmittersIndexes[i];

    __syncthreads();

    for(int t = 0; t < transmittersCount; ++t) {
        int currTransmitterIndex = cachedTransmittersIndexes[t];
        float realReceiver = CudaCommonsUtils::getRealReceiver(currTransmitterIndex, receiversCount, receiverWidth);
        const int inputOffset = receiversCount * samplesCount * t;
        float transmitDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, realReceiver);

        result += OneTransmitSta<apod, useTexture, T>()(input, receiversCount, samplesCount, receiverWidth,
                                                        soundVelocity, samplingFrequency, startDepth, hanningWindow,
                                                        inputOffset,
                                                        transmitDistance, realX, realY, realReceiver,
                                                        transmitFrequency);
    }

    output[x + y * blockDim.x * gridDim.x] = CudaCommonsUtils::getComplexModulus(result);
}

template<STA_APODIZATION apod, bool useTexture, typename T>
__global__ void gpuStaWithFocusing(const T *input, float *output, const int widthInPixels, const int heightInPixels,
                                   const float areaHeight, const int receiversCount,
                                   const int samplesCount, const float receiverWidth, const float soundVelocity,
                                   const float samplingFrequency, const float startDepth,
                                   const float *hanningWindow, const float transmitFrequency) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float realX = (float) x / (float) (widthInPixels - 1) * receiverWidth;
    float realY = (float) y / (float) (heightInPixels - 1) * areaHeight + startDepth;

    const int inputOffset = receiversCount * samplesCount * x;
    float transmitDistance = realY;
    float realReceiver = CudaCommonsUtils::getRealReceiver(x, receiversCount, receiverWidth);
    T result = OneTransmitSta<apod, useTexture, T>()(input, receiversCount, samplesCount, receiverWidth, soundVelocity,
                                                     samplingFrequency, startDepth, hanningWindow, inputOffset,
                                                     transmitDistance, realX, realY, realReceiver, transmitFrequency);

    output[x + y * blockDim.x * gridDim.x] = CudaCommonsUtils::getComplexModulus(result);
}

void CudaStaGraphNode::staRf(const float *inputPtr, float *outputPtr, cudaStream_t &stream, const int width,
                             const int height, const int receiversCount,
                             const int samplesCount, const float receiverWidth, const float soundVelocity,
                             const float samplingFrequency, const float startDepth,
                             const float *transmittersIndexes, const int transmittersCount, const float *hanningWindow,
                             const float transmitFrequency, const STA_APODIZATION apod,
                             const CudaStaPixelMap pixelMap) {
    this->loadCudaDeviceProps();
    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texStaRfInputData, inputPtr,
                                                             samplesCount * receiversCount * transmittersCount,
                                                             this->deviceProp.get());

    dim3 blockDim(8, 32);
    dim3 gridDim(width / blockDim.x, height / blockDim.y);
    int externalSharedMemory = sizeof(int) * transmittersCount;
    if(isTextureBinded) {
        if(apod == STA_APODIZATION::NONE)
            gpuSta<STA_APODIZATION::NONE, true> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        else if(apod == STA_APODIZATION::TAS)
            gpuSta<STA_APODIZATION::TAS, true> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        else if(apod == STA_APODIZATION::HANN)
            gpuSta<STA_APODIZATION::HANN, true> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        CUDA_ASSERT(cudaUnbindTexture(&texStaRfInputData));
    } else {
        if(apod == STA_APODIZATION::NONE)
            gpuSta<STA_APODIZATION::NONE, false> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        else if(apod == STA_APODIZATION::TAS)
            gpuSta<STA_APODIZATION::TAS, false> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        else if(apod == STA_APODIZATION::HANN)
            gpuSta<STA_APODIZATION::HANN, false> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
    }
    CUDA_ASSERT(cudaGetLastError());
}

void CudaStaGraphNode::staIq(const float2 *inputPtr, float *outputPtr, cudaStream_t &stream, const int width,
                             const int height, const int receiversCount,
                             const int samplesCount, const float receiverWidth, const float soundVelocity,
                             const float samplingFrequency, const float startDepth,
                             const float *transmittersIndexes, const int transmittersCount, const float *hanningWindow,
                             const float transmitFrequency, const STA_APODIZATION apod,
                             const CudaStaPixelMap pixelMap) {
    this->loadCudaDeviceProps();
    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texStaIqInputData, inputPtr,
                                                             samplesCount * receiversCount * transmittersCount,
                                                             this->deviceProp.get());

    dim3 blockDim(8, 32);
    dim3 gridDim(width / blockDim.x, height / blockDim.y);
    int externalSharedMemory = sizeof(int) * transmittersCount;
    if(isTextureBinded) {
        if(apod == STA_APODIZATION::NONE)
            gpuSta<STA_APODIZATION::NONE, true> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        else if(apod == STA_APODIZATION::TAS)
            gpuSta<STA_APODIZATION::TAS, true> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        else if(apod == STA_APODIZATION::HANN)
            gpuSta<STA_APODIZATION::HANN, true> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        CUDA_ASSERT(cudaUnbindTexture(&texStaIqInputData));
    } else {
        if(apod == STA_APODIZATION::NONE)
            gpuSta<STA_APODIZATION::NONE, false> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        else if(apod == STA_APODIZATION::TAS)
            gpuSta<STA_APODIZATION::TAS, false> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
        else if(apod == STA_APODIZATION::HANN)
            gpuSta<STA_APODIZATION::HANN, false> << <
            gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
            transmittersIndexes, transmittersCount, hanningWindow, transmitFrequency,
            pixelMap);
    }
    CUDA_ASSERT(cudaGetLastError());
}

void CudaStaGraphNode::staRfWithFocusing(const float *inputPtr, float *outputPtr, cudaStream_t &stream, const int width,
                                         const int height, const float areaHeight, const int receiversCount,
                                         const int samplesCount, const float receiverWidth, const float soundVelocity,
                                         const float samplingFrequency, const float startDepth,
                                         const float *hanningWindow, const float transmitFrequency,
                                         const STA_APODIZATION apod) {
    this->loadCudaDeviceProps();
    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texStaRfInputData, inputPtr,
                                                             samplesCount * receiversCount * receiversCount,
                                                             this->deviceProp.get());

    dim3 blockDim(1, std::min(512, height));
    dim3 gridDim(width / blockDim.x, height / blockDim.y);
    if(isTextureBinded) {
        if(apod == STA_APODIZATION::NONE)
            gpuStaWithFocusing<STA_APODIZATION::NONE, true> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        else if(apod == STA_APODIZATION::TAS)
            gpuStaWithFocusing<STA_APODIZATION::TAS, true> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        else if(apod == STA_APODIZATION::HANN)
            gpuStaWithFocusing<STA_APODIZATION::HANN, true> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        CUDA_ASSERT(cudaUnbindTexture(&texStaRfInputData));
    } else {
        if(apod == STA_APODIZATION::NONE)
            gpuStaWithFocusing<STA_APODIZATION::NONE, false> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        else if(apod == STA_APODIZATION::TAS)
            gpuStaWithFocusing<STA_APODIZATION::TAS, false> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        else if(apod == STA_APODIZATION::HANN)
            gpuStaWithFocusing<STA_APODIZATION::HANN, false> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
    }
    CUDA_ASSERT(cudaGetLastError());
}

void
CudaStaGraphNode::staIqWithFocusing(const float2 *inputPtr, float *outputPtr, cudaStream_t &stream, const int width,
                                    const int height, const float areaHeight, const int receiversCount,
                                    const int samplesCount, const float receiverWidth, const float soundVelocity,
                                    const float samplingFrequency, const float startDepth,
                                    const float *hanningWindow, const float transmitFrequency,
                                    const STA_APODIZATION apod) {
    this->loadCudaDeviceProps();
    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texStaIqInputData, inputPtr,
                                                             samplesCount * receiversCount * receiversCount,
                                                             this->deviceProp.get());

    dim3 blockDim(1, std::min(512, height));
    dim3 gridDim(width / blockDim.x, height / blockDim.y);
    if(isTextureBinded) {
        if(apod == STA_APODIZATION::NONE)
            gpuStaWithFocusing<STA_APODIZATION::NONE, true> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        else if(apod == STA_APODIZATION::TAS)
            gpuStaWithFocusing<STA_APODIZATION::TAS, true> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        else if(apod == STA_APODIZATION::HANN)
            gpuStaWithFocusing<STA_APODIZATION::HANN, true> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        CUDA_ASSERT(cudaUnbindTexture(&texStaIqInputData));
    } else {
        if(apod == STA_APODIZATION::NONE)
            gpuStaWithFocusing<STA_APODIZATION::NONE, false> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        else if(apod == STA_APODIZATION::TAS)
            gpuStaWithFocusing<STA_APODIZATION::TAS, false> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
        else if(apod == STA_APODIZATION::HANN)
            gpuStaWithFocusing<STA_APODIZATION::HANN, false> << <
            gridDim, blockDim, 0, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount,
            samplesCount, receiverWidth, soundVelocity, samplingFrequency,
            startDepth, hanningWindow, transmitFrequency);
    }
    CUDA_ASSERT(cudaGetLastError());
}

void CudaStaGraphNode::loadCudaDeviceProps() {
    if(!this->deviceProp) {
        this->deviceProp = boost::optional<cudaDeviceProp>(CudaApiUtils::getCudaDeviceProps());
    }
}