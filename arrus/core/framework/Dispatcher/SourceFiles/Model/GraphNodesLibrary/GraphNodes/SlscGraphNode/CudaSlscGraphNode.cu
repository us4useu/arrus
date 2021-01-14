#include "Model/GraphNodesLibrary/GraphNodes/SlscGraphNode/CudaSlscGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaCommonsUtils.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaApiUtils.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaVectorMathUtils.cuh"
#include <algorithm>

texture<float, cudaTextureType1D, cudaReadModeElementType> texSlscRfInputData;
texture <float2, cudaTextureType1D, cudaReadModeElementType> texSlscIqInputData;

template<bool useTexture, typename T>
__device__ __forceinline__ T getInputDataValue(const T *inputData, const int sampleOffset) {
    return inputData[sampleOffset];
}

template<>
__device__ __forceinline__ float getInputDataValue<true, float>(const float *inputData, const int sampleOffset) {
    return tex1Dfetch(texSlscRfInputData, sampleOffset);
}

template<>
__device__ __forceinline__ float2 getInputDataValue<true, float2>(const float2 *inputData, const int sampleOffset) {
    return tex1Dfetch(texSlscIqInputData, sampleOffset);
}

template<bool useTexture>
__device__ __forceinline__ float
calculateResultForFirstReceiverRf(const float *input, const int inputOffset, const float realX,
                                  const float transmitDistance, const float realY, const int receiversCount,
                                  const int samplesCount, const float receiverWidth, const float soundVelocity,
                                  const float samplingFrequency, const float startDepth,
                                  const int offset, float *cachedPartialResults, const int maxLag) {
    float result = 0.0f;

    int localIndex = (threadIdx.x + threadIdx.y * blockDim.x) * (2 * offset + 1);

    float receiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, 0, receiversCount, receiverWidth);
    int sample = (receiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;

    // calculate result for first receiver for maxLag
    for(int lag = maxLag; lag > 0; --lag) {
        float nominator = 0.0f;
        float denominatorA = 0.0f;
        float denominatorB = 0.0f;

        float secondReceiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, lag, receiversCount,
                                                                           receiverWidth);
        int secondSample =
            (secondReceiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;

        // signals correlation
        for(int i = -offset; i <= offset; ++i) {
            int tempSample = sample + i;
            int tempSample2 = secondSample + i;
            float firstValue = 0.0f, secondValue = 0.0f;
            if((tempSample < samplesCount) && (tempSample >= 0))
                firstValue = getInputDataValue<useTexture>(input, inputOffset + tempSample);
            if((tempSample2 < samplesCount) && (tempSample2 >= 0))
                secondValue = getInputDataValue<useTexture>(input, inputOffset + tempSample2 + lag * samplesCount);

            nominator += firstValue * secondValue;
            denominatorA += firstValue * firstValue;
            denominatorB += secondValue * secondValue;
        }

        // normalization
        if(denominatorA * denominatorB != 0.0f)
            result += nominator / sqrt(denominatorA * denominatorB);

        if(denominatorB != 0.0f) {
            denominatorB = 1.0f / sqrt(denominatorB);
            for(int i = -offset, li = 0; i <= offset; ++i, ++li) {
                int tempSample2 = secondSample + i;
                if((tempSample2 < samplesCount) && (tempSample2 >= 0)) {
                    float secondValue = getInputDataValue<useTexture>(input,
                                                                      inputOffset + tempSample2 + lag * samplesCount);

                    // cache partial result for one probe in correlated signal
                    cachedPartialResults[localIndex + li] += secondValue * denominatorB;
                }
            }
        }
    }

    return result;
}

template<bool useTexture>
__device__ __forceinline__ float
calculateResultForMiddleReceiversRf(const float *input, const int inputOffset, const float realX,
                                    const float transmitDistance, const float realY, const int receiversCount,
                                    const int samplesCount, const float receiverWidth, const float soundVelocity,
                                    const float samplingFrequency, const float startDepth,
                                    const int offset, float *cachedPartialResults, const int maxLag) {
    float result = 0.0f;

    int localIndex = (threadIdx.x + threadIdx.y * blockDim.x) * (2 * offset + 1);

    // calculate result for most of rest receivers based on the result from first receiver (without borders receivers)
    for(int r = 1; r < receiversCount - maxLag; ++r) {
        // old receiver
        float receiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, r, receiversCount, receiverWidth);
        int sample = (receiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;
        // new receiver
        float secondReceiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, r + maxLag, receiversCount,
                                                                           receiverWidth);
        int secondSample =
            (secondReceiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;

        float oldReceiversValuesSquare = 0.0f;
        float newReceiversValuesSquare = 0.0f;

        // calculate values squares
        for(int i = -offset; i <= offset; ++i) {
            int tempSample = sample + i;
            int tempSample2 = secondSample + i;
            if((tempSample < samplesCount) && (tempSample >= 0)) {
                float firstValue = getInputDataValue<useTexture>(input, inputOffset + tempSample + r * samplesCount);
                oldReceiversValuesSquare += firstValue * firstValue;
            }
            if((tempSample2 < samplesCount) && (tempSample2 >= 0)) {
                float secondValue = getInputDataValue<useTexture>(input, inputOffset + tempSample2 +
                                                                         (r + maxLag) * samplesCount);
                newReceiversValuesSquare += secondValue * secondValue;
            }
        }

        oldReceiversValuesSquare = (oldReceiversValuesSquare == 0.0f) ? 0.0f : 1.0f / sqrt(oldReceiversValuesSquare);
        newReceiversValuesSquare = (newReceiversValuesSquare == 0.0f) ? 0.0f : 1.0f / sqrt(newReceiversValuesSquare);
        // signals correlation
        for(int i = -offset, li = 0; i <= offset; ++i, ++li) {
            int tempSample = sample + i;
            int tempSample2 = secondSample + i;
            float b1b2 = 0.0f, e1e2 = 0.0f;
            if((tempSample < samplesCount) && (tempSample >= 0)) {
                float firstValue = getInputDataValue<useTexture>(input, inputOffset + tempSample + r * samplesCount);
                b1b2 = firstValue * oldReceiversValuesSquare;
            }
            if((tempSample2 < samplesCount) && (tempSample2 >= 0)) {
                float secondValue = getInputDataValue<useTexture>(input, inputOffset + tempSample2 +
                                                                         (r + maxLag) * samplesCount);
                e1e2 = secondValue * newReceiversValuesSquare;
            }
            float xn = cachedPartialResults[localIndex + li];
            xn = xn - b1b2 + e1e2;
            cachedPartialResults[localIndex + li] = xn;
            result += xn * b1b2;
        }
    }

    return result;
}

template<bool useTexture>
__device__ __forceinline__ float
calculateResultForBordersReceiversRf(const float *input, const int inputOffset, const float realX,
                                     const float transmitDistance, const float realY, const int receiversCount,
                                     const int samplesCount, const float receiverWidth, const float soundVelocity,
                                     const float samplingFrequency, const float startDepth,
                                     const int offset, float *cachedPartialResults, const int maxLag) {
    float result = 0.0f;

    int localIndex = (threadIdx.x + threadIdx.y * blockDim.x) * (2 * offset + 1);

    // calculate result for borders receivers
    for(int r = receiversCount - maxLag; r < receiversCount; ++r) {
        // old receiver
        float receiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, r, receiversCount, receiverWidth);
        int sample = (receiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;

        float oldReceiverValuesSquare = 0.0f;

        // calculate values squares
        for(int i = -offset; i <= offset; ++i) {
            int tempSample = sample + i;
            if((tempSample < samplesCount) && (tempSample >= 0)) {
                float firstValue = getInputDataValue<useTexture>(input, inputOffset + tempSample + r * samplesCount);
                oldReceiverValuesSquare += firstValue * firstValue;
            }
        }

        oldReceiverValuesSquare = (oldReceiverValuesSquare == 0.0f) ? 0.0f : 1.0f / sqrt(oldReceiverValuesSquare);
        // signals correlation
        for(int i = -offset, li = 0; i <= offset; ++i, ++li) {
            int tempSample = sample + i;
            if((tempSample < samplesCount) && (tempSample >= 0)) {
                float firstValue = getInputDataValue<useTexture>(input, inputOffset + tempSample + r * samplesCount);

                float xn = cachedPartialResults[localIndex + li];
                float b1b2 = firstValue * oldReceiverValuesSquare;
                xn = xn - b1b2;
                cachedPartialResults[localIndex + li] = xn;
                result += xn * b1b2;
            }
        }
    }

    return result;
}

template<bool useTexture>
__device__ __forceinline__ float
calculateResultForFirstReceiverIq(const float2 *input, const int inputOffset, const float realX,
                                  const float transmitDistance, const float realY, const int receiversCount,
                                  const int samplesCount, const float receiverWidth, const float soundVelocity,
                                  const float samplingFrequency, const float startDepth,
                                  float *cachedPartialResults, const int maxLag, const float transmitFrequency) {
    float result = 0.0f;
    int localIndex = (threadIdx.x + threadIdx.y * blockDim.x) * 2;

    float receiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, 0, receiversCount, receiverWidth);
    int sample = (receiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;

    float2 sinCosFirst;
    __sincosf(-2.0f * CUDART_PI_F * transmitFrequency * (receiveDistance + transmitDistance - startDepth * 2.0f) /
              soundVelocity, &sinCosFirst.x, &sinCosFirst.y);

    float2 firstValueIq = make_float2(0.0f);
    if((sample < samplesCount) && (sample >= 0))
        firstValueIq = getInputDataValue<useTexture>(input, inputOffset + sample);

    float2 firstValue;
    firstValue.x = firstValueIq.x * sinCosFirst.y - firstValueIq.y * sinCosFirst.x;
    firstValue.y = firstValueIq.x * sinCosFirst.x + firstValueIq.y * sinCosFirst.y;

    // calculate result for first receiver for maxLag
    for(int lag = maxLag; lag > 0; --lag) {
        float secondReceiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, lag, receiversCount,
                                                                           receiverWidth);
        int secondSample =
            (secondReceiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;
        float2 sinCosSecond;
        __sincosf(
            -2.0f * CUDART_PI_F * transmitFrequency * (secondReceiveDistance + transmitDistance - startDepth * 2.0f) /
            soundVelocity, &sinCosSecond.x, &sinCosSecond.y);

        float2 secondValueIq = make_float2(0.0f);
        if((secondSample < samplesCount) && (secondSample >= 0))
            secondValueIq = getInputDataValue<useTexture>(input, inputOffset + secondSample + lag * samplesCount);

        float2 secondValue;
        secondValue.x = secondValueIq.x * sinCosSecond.y - secondValueIq.y * sinCosSecond.x;
        secondValue.y = secondValueIq.x * sinCosSecond.x + secondValueIq.y * sinCosSecond.y;

        float nominator = firstValue.x * secondValue.x + firstValue.y * secondValue.y;
        float denominatorA = firstValue.x * firstValue.x + firstValue.y * firstValue.y;
        float denominatorB = secondValue.x * secondValue.x + secondValue.y * secondValue.y;

        // normalization
        if(denominatorA * denominatorB != 0.0f)
            result += nominator / sqrt(denominatorA * denominatorB);

        if(denominatorB != 0.0f) {
            denominatorB = 1.0f / sqrt(denominatorB);
            // cache partial result
            cachedPartialResults[localIndex] += secondValue.x * denominatorB;
            cachedPartialResults[localIndex + 1] += secondValue.y * denominatorB;
        }
    }

    return result;
}

template<bool useTexture>
__device__ __forceinline__ float
calculateResultForMiddleReceiversIq(const float2 *input, const int inputOffset, const float realX,
                                    const float transmitDistance, const float realY, const int receiversCount,
                                    const int samplesCount, const float receiverWidth, const float soundVelocity,
                                    const float samplingFrequency, const float startDepth,
                                    float *cachedPartialResults, const int maxLag, const float transmitFrequency) {
    float result = 0.0f;
    int localIndex = (threadIdx.x + threadIdx.y * blockDim.x) * 2;

    // calculate result for most of rest receivers based on the result from first receiver (without borders receivers)
    for(int r = 1; r < receiversCount - maxLag; ++r) {
        // old receiver
        float receiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, r, receiversCount, receiverWidth);
        int sample = (receiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;
        float2 sinCosFirst;
        __sincosf(-2.0f * CUDART_PI_F * transmitFrequency * (receiveDistance + transmitDistance - startDepth * 2.0f) /
                  soundVelocity, &sinCosFirst.x, &sinCosFirst.y);

        float2 firstValueIq = make_float2(0.0f);
        if((sample < samplesCount) && (sample >= 0))
            firstValueIq = getInputDataValue<useTexture>(input, inputOffset + sample + r * samplesCount);

        float2 firstValue;
        firstValue.x = firstValueIq.x * sinCosFirst.y - firstValueIq.y * sinCosFirst.x;
        firstValue.y = firstValueIq.x * sinCosFirst.x + firstValueIq.y * sinCosFirst.y;

        float oldReceiversValuesSquare = firstValue.x * firstValue.x + firstValue.y * firstValue.y;
        oldReceiversValuesSquare = (oldReceiversValuesSquare == 0.0f) ? 0.0f : 1.0f / sqrt(oldReceiversValuesSquare);
        float2 b1b2 = firstValue * oldReceiversValuesSquare;

        // new receiver
        float secondReceiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, r + maxLag, receiversCount,
                                                                           receiverWidth);
        int secondSample =
            (secondReceiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;
        float2 sinCosSecond;
        __sincosf(
            -2.0f * CUDART_PI_F * transmitFrequency * (secondReceiveDistance + transmitDistance - startDepth * 2.0f) /
            soundVelocity, &sinCosSecond.x, &sinCosSecond.y);

        float2 secondValueIq = make_float2(0.0f);
        if((secondSample < samplesCount) && (secondSample >= 0))
            secondValueIq = getInputDataValue<useTexture>(input,
                                                          inputOffset + secondSample + (r + maxLag) * samplesCount);

        float2 secondValue;
        secondValue.x = secondValueIq.x * sinCosSecond.y - secondValueIq.y * sinCosSecond.x;
        secondValue.y = secondValueIq.x * sinCosSecond.x + secondValueIq.y * sinCosSecond.y;

        float newReceiversValuesSquare = secondValue.x * secondValue.x + secondValue.y * secondValue.y;
        newReceiversValuesSquare = (newReceiversValuesSquare == 0.0f) ? 0.0f : 1.0f / sqrt(newReceiversValuesSquare);
        float2 e1e2 = secondValue * newReceiversValuesSquare;

        float2 xn = make_float2(cachedPartialResults[localIndex], cachedPartialResults[localIndex + 1]);
        xn = xn - b1b2 + e1e2;
        cachedPartialResults[localIndex] = xn.x;
        cachedPartialResults[localIndex + 1] = xn.y;
        result += xn.x * b1b2.x + xn.y * b1b2.y;
    }

    return result;
}

template<bool useTexture>
__device__ __forceinline__ float
calculateResultForBordersReceiversIq(const float2 *input, const int inputOffset, const float realX,
                                     const float transmitDistance, const float realY, const int receiversCount,
                                     const int samplesCount, const float receiverWidth, const float soundVelocity,
                                     const float samplingFrequency, const float startDepth,
                                     float *cachedPartialResults, const int maxLag, const float transmitFrequency) {
    float result = 0.0f;
    int localIndex = (threadIdx.x + threadIdx.y * blockDim.x) * 2;

    // calculate result for borders receivers
    for(int r = receiversCount - maxLag; r < receiversCount; ++r) {
        // old receiver
        float receiveDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, r, receiversCount, receiverWidth);
        int sample = (receiveDistance + transmitDistance - startDepth * 2.0f) / soundVelocity * samplingFrequency;
        float2 sinCosFirst;
        __sincosf(-2.0f * CUDART_PI_F * transmitFrequency * (receiveDistance + transmitDistance - startDepth * 2.0f) /
                  soundVelocity, &sinCosFirst.x, &sinCosFirst.y);

        float2 firstValueIq = make_float2(0.0f);

        // calculate values squares
        if((sample < samplesCount) && (sample >= 0))
            firstValueIq = getInputDataValue<useTexture>(input, inputOffset + sample + r * samplesCount);

        float2 firstValue;
        firstValue.x = firstValueIq.x * sinCosFirst.y - firstValueIq.y * sinCosFirst.x;
        firstValue.y = firstValueIq.x * sinCosFirst.x + firstValueIq.y * sinCosFirst.y;

        float oldReceiverValuesSquare = firstValue.x * firstValue.x + firstValue.y * firstValue.y;
        oldReceiverValuesSquare = (oldReceiverValuesSquare == 0.0f) ? 0.0f : 1.0f / sqrt(oldReceiverValuesSquare);

        // signals correlation
        float2 xn = make_float2(cachedPartialResults[localIndex], cachedPartialResults[localIndex + 1]);
        float2 b1b2 = firstValue * oldReceiverValuesSquare;
        xn = xn - b1b2;
        cachedPartialResults[localIndex] = xn.x;
        cachedPartialResults[localIndex + 1] = xn.y;
        result += xn.x * b1b2.x + xn.y * b1b2.y;
    }

    return result;
}

template<bool useTexture, typename T>
struct SlscCommonSpec {
    __device__ __forceinline__ float
    operator()(const T *input, const int inputOffset, const float realX, const float transmitDistance,
               const float realY, const int receiversCount,
               const int samplesCount, const float receiverWidth, const float soundVelocity,
               const float samplingFrequency, const float startDepth,
               const int lags, const int offset, float *cachedPartialResults, const float transmitFrequency) {
        int localIndex = (threadIdx.x + threadIdx.y * blockDim.x) * (2 * offset + 1);

        // clear variables for partial lags results
        for(int i = -offset, li = 0; i <= offset; ++i, ++li)
            cachedPartialResults[localIndex + li] = 0.0f;

        float result = 0.0f;
        result += calculateResultForFirstReceiverRf<useTexture>(input, inputOffset, realX, transmitDistance, realY,
                                                                receiversCount, samplesCount, receiverWidth,
                                                                soundVelocity, samplingFrequency,
                                                                startDepth, offset, cachedPartialResults, lags);

        result += calculateResultForMiddleReceiversRf<useTexture>(input, inputOffset, realX, transmitDistance, realY,
                                                                  receiversCount, samplesCount, receiverWidth,
                                                                  soundVelocity, samplingFrequency,
                                                                  startDepth, offset, cachedPartialResults, lags);

        result += calculateResultForBordersReceiversRf<useTexture>(input, inputOffset, realX, transmitDistance, realY,
                                                                   receiversCount, samplesCount, receiverWidth,
                                                                   soundVelocity, samplingFrequency,
                                                                   startDepth, offset, cachedPartialResults, lags);
        return result;
    }
};

template<bool useTexture>
struct SlscCommonSpec<useTexture, float2> {
    __device__ __forceinline__ float
    operator()(const float2 *input, const int inputOffset, const float realX, const float transmitDistance,
               const float realY, const int receiversCount,
               const int samplesCount, const float receiverWidth, const float soundVelocity,
               const float samplingFrequency, const float startDepth,
               const int lags, const int offset, float *cachedPartialResults, const float transmitFrequency) {
        int localIndex = (threadIdx.x + threadIdx.y * blockDim.x) * 2;
        cachedPartialResults[localIndex] = 0.0f;
        cachedPartialResults[localIndex + 1] = 0.0f;

        float result = 0.0f;
        result += calculateResultForFirstReceiverIq<useTexture>(input, inputOffset, realX, transmitDistance, realY,
                                                                receiversCount, samplesCount, receiverWidth,
                                                                soundVelocity, samplingFrequency,
                                                                startDepth, cachedPartialResults, lags,
                                                                transmitFrequency);

        result += calculateResultForMiddleReceiversIq<useTexture>(input, inputOffset, realX, transmitDistance, realY,
                                                                  receiversCount, samplesCount, receiverWidth,
                                                                  soundVelocity, samplingFrequency,
                                                                  startDepth, cachedPartialResults, lags,
                                                                  transmitFrequency);

        result += calculateResultForBordersReceiversIq<useTexture>(input, inputOffset, realX, transmitDistance, realY,
                                                                   receiversCount, samplesCount, receiverWidth,
                                                                   soundVelocity, samplingFrequency,
                                                                   startDepth, cachedPartialResults, lags,
                                                                   transmitFrequency);
        return result;
    }
};

template<bool useTexture, typename T>
__device__ __forceinline__ float
slscCommon(const T *input, const int inputOffset, const float realX, const float transmitDistance, const float realY,
           const int receiversCount,
           const int samplesCount, const float receiverWidth, const float soundVelocity, const float samplingFrequency,
           const float startDepth,
           const int lags, const int offset, float *cachedPartialResults, const float areaHeight,
           const float transmitFrequency) {
    // calculate more lags for deeper part of the picture
    int maxLag = lags;
    //maxLag = fminf(1 + realY / (areaHeight * 0.5f) * (float)(lags - 1), lags);

    float result = SlscCommonSpec<useTexture, T>()(input, inputOffset, realX, transmitDistance, realY, receiversCount,
                                                   samplesCount, receiverWidth, soundVelocity, samplingFrequency,
                                                   startDepth, maxLag, offset, cachedPartialResults, transmitFrequency);

    return result;// *((float)lags / (float)maxLag);
}

template<bool useTexture, typename T>
__global__ void
gpuSlsc(const T *input, float *output, const int widthInPixels, const int heightInPixels, const float areaHeight,
        const int receiversCount,
        const int samplesCount, const float receiverWidth, const float soundVelocity, const float samplingFrequency,
        const float startDepth,
        const int *transmittersIndexes, const int transmittersCount, const int lags, const int offset,
        const float transmitFrequency, const CudaSlscPixelMap pixelMap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float realX = (float) x / (float) (widthInPixels - 1) * pixelMap.width + pixelMap.x;
    float realY = (float) y / (float) (heightInPixels - 1) * pixelMap.height + pixelMap.y;

    float result = 0.0f;

    extern __shared__ int cachedTransmittersIndexes[];

    int localIdx = threadIdx.x + threadIdx.y * blockDim.x;
    for(int i = localIdx; i < transmittersCount; i += blockDim.x * blockDim.y)
        cachedTransmittersIndexes[i] = transmittersIndexes[i];

    __syncthreads();

    float *cachedPartialResults = (float *) &cachedTransmittersIndexes[transmittersCount];

    for(int t = 0; t < transmittersCount; ++t) {
        const int inputOffset = receiversCount * samplesCount * t;
        int currTransmitterIndex = cachedTransmittersIndexes[t];
        float transmitDistance = CudaCommonsUtils::distanceToReceiver(realX, realY, currTransmitterIndex,
                                                                      receiversCount, receiverWidth);
        result += slscCommon<useTexture>(input, inputOffset, realX, transmitDistance, realY, receiversCount,
                                         samplesCount, receiverWidth, soundVelocity,
                                         samplingFrequency, startDepth, lags, offset, cachedPartialResults, areaHeight,
                                         transmitFrequency);
    }

    output[x + y * blockDim.x * gridDim.x] = result;
}

template<bool useTexture, typename T>
__global__ void gpuSlscWithFocusing(const T *input, float *output, const int widthInPixels, const int heightInPixels,
                                    const float areaHeight, const int receiversCount,
                                    const int samplesCount, const float receiverWidth, const float soundVelocity,
                                    const float samplingFrequency,
                                    const float startDepth, const int lags, const int offset,
                                    const float transmitFrequency) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float realX = (float) x / (float) (widthInPixels - 1) * receiverWidth;
    float realY = (float) y / (float) (heightInPixels - 1) * areaHeight + startDepth;

    extern __shared__ float cachedPartialResults[];

    const int inputOffset = receiversCount * samplesCount * x;
    float transmitDistance = realY;
    float result = slscCommon<useTexture>(input, inputOffset, realX, transmitDistance, realY, receiversCount,
                                          samplesCount, receiverWidth, soundVelocity, samplingFrequency,
                                          startDepth, lags, offset, cachedPartialResults, areaHeight,
                                          transmitFrequency);

    output[x + y * blockDim.x * gridDim.x] = result;
}

template<typename T>
__global__ void gpuRemoveConstant(T *input, const int samplesCount) {
    int x = threadIdx.x;
    int offset = blockIdx.x * samplesCount;
    extern __shared__ float extData[];
    T *data = (T *) extData;

    data[x] = makeZeroValue<T>();
    for(int i = x; i < samplesCount; i += blockDim.x)
        data[x] += input[offset + i];

    __syncthreads();

    for(int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if(x < i) {
            T a = data[x];
            T b = data[x + i];
            data[x] = a + b;
        }
        __syncthreads();
    }

    T mean = data[0] / (float) samplesCount;

    for(int i = x; i < samplesCount; i += blockDim.x)
        input[offset + i] -= mean;
}

void CudaSlscGraphNode::removeConstantRf(float *inputPtr, const int samplesCount, const int signalsCount,
                                         const cudaStream_t &stream) {
    dim3 removeBlockDim(512);
    dim3 removeGridDim(signalsCount);
    int externalSharedMemory = sizeof(float) * removeBlockDim.x;
    gpuRemoveConstant << < removeGridDim, removeBlockDim, externalSharedMemory, stream >> >(inputPtr, samplesCount);
    CUDA_ASSERT(cudaGetLastError());
}

void CudaSlscGraphNode::removeConstantIq(float2 *inputPtr, const int samplesCount, const int signalsCount,
                                         const cudaStream_t &stream) {
    dim3 removeBlockDim(512);
    dim3 removeGridDim(signalsCount);
    int externalSharedMemory = sizeof(float2) * removeBlockDim.x;
    gpuRemoveConstant << < removeGridDim, removeBlockDim, externalSharedMemory, stream >> >(inputPtr, samplesCount);
    CUDA_ASSERT(cudaGetLastError());
}

void CudaSlscGraphNode::slscRf(float *inputPtr, float *outputPtr, const cudaStream_t &stream, const int width,
                               const int height, const float areaHeight, const int receiversCount,
                               const int samplesCount, const float receiverWidth, const float soundVelocity,
                               const float samplingFrequency, const float startDepth,
                               const int *transmittersIndexes, const int transmittersCount, const int lags,
                               const int offset, const CudaSlscPixelMap pixelMap) {
    this->loadCudaDeviceProps();
    this->removeConstantRf(inputPtr, samplesCount, transmittersCount * receiversCount, stream);

    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texSlscRfInputData, inputPtr,
                                                             samplesCount * receiversCount * transmittersCount,
                                                             this->deviceProp.get());

    dim3 blockDim(8, 32);
    dim3 gridDim(width / blockDim.x, height / blockDim.y);
    int externalSharedMemory =
        sizeof(int) * transmittersCount + sizeof(float) * blockDim.x * blockDim.y * (offset * 2 + 1);
    if(isTextureBinded) {
        gpuSlsc<true> << <
        gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount, samplesCount, receiverWidth,
            soundVelocity, samplingFrequency, startDepth, transmittersIndexes, transmittersCount,
            lags, offset, 0.0f, pixelMap);
        CUDA_ASSERT(cudaUnbindTexture(&texSlscRfInputData));
    } else {
        gpuSlsc<false> << <
        gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount, samplesCount, receiverWidth,
            soundVelocity, samplingFrequency, startDepth, transmittersIndexes, transmittersCount,
            lags, offset, 0.0f, pixelMap);
    }

    CUDA_ASSERT(cudaGetLastError());
}

void
CudaSlscGraphNode::slscWithFocusingRf(float *inputPtr, float *outputPtr, const cudaStream_t &stream, const int width,
                                      const int height, const float areaHeight, const int receiversCount,
                                      const int samplesCount, const float receiverWidth, const float soundVelocity,
                                      const float samplingFrequency, const float startDepth,
                                      const int lags, const int offset) {
    this->loadCudaDeviceProps();
    this->removeConstantRf(inputPtr, samplesCount, receiversCount * receiversCount, stream);

    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texSlscRfInputData, inputPtr,
                                                             samplesCount * receiversCount * receiversCount,
                                                             this->deviceProp.get());

    dim3 blockDim(1, std::min(512, height));
    dim3 gridDim(width / blockDim.x, height / blockDim.y);
    int externalSharedMemory = sizeof(float) * blockDim.x * blockDim.y * (offset * 2 + 1);
    if(isTextureBinded) {
        gpuSlscWithFocusing<true> << <
        gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount, samplesCount, receiverWidth,
            soundVelocity, samplingFrequency, startDepth, lags, offset, 0.0f);
        CUDA_ASSERT(cudaUnbindTexture(&texSlscRfInputData));
    } else {
        gpuSlscWithFocusing<false> << <
        gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount, samplesCount, receiverWidth,
            soundVelocity, samplingFrequency, startDepth, lags, offset, 0.0f);
    }

    CUDA_ASSERT(cudaGetLastError());
}

void CudaSlscGraphNode::slscIq(float2 *inputPtr, float *outputPtr, const cudaStream_t &stream, const int width,
                               const int height, const float areaHeight, const int receiversCount,
                               const int samplesCount, const float receiverWidth, const float soundVelocity,
                               const float samplingFrequency, const float startDepth,
                               const int *transmittersIndexes, const int transmittersCount, const int lags,
                               const float transmitFrequency, const CudaSlscPixelMap pixelMap) {
    this->loadCudaDeviceProps();
    this->removeConstantIq(inputPtr, samplesCount, transmittersCount * receiversCount, stream);

    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texSlscIqInputData, inputPtr,
                                                             samplesCount * receiversCount * transmittersCount,
                                                             this->deviceProp.get());

    dim3 blockDim(8, 32);
    dim3 gridDim(width / blockDim.x, height / blockDim.y);
    int externalSharedMemory = sizeof(int) * transmittersCount + sizeof(float) * blockDim.x * blockDim.y * 2;
    if(isTextureBinded) {
        gpuSlsc<true> << <
        gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount, samplesCount, receiverWidth,
            soundVelocity, samplingFrequency, startDepth, transmittersIndexes, transmittersCount,
            lags, 0, transmitFrequency, pixelMap);
        CUDA_ASSERT(cudaUnbindTexture(&texSlscIqInputData));
    } else {
        gpuSlsc<false> << <
        gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount, samplesCount, receiverWidth,
            soundVelocity, samplingFrequency, startDepth, transmittersIndexes, transmittersCount,
            lags, 0, transmitFrequency, pixelMap);
    }

    CUDA_ASSERT(cudaGetLastError());
}

void
CudaSlscGraphNode::slscWithFocusingIq(float2 *inputPtr, float *outputPtr, const cudaStream_t &stream, const int width,
                                      const int height, const float areaHeight, const int receiversCount,
                                      const int samplesCount, const float receiverWidth, const float soundVelocity,
                                      const float samplingFrequency,
                                      const float startDepth, const int lags, const float transmitFrequency) {
    this->loadCudaDeviceProps();
    this->removeConstantIq(inputPtr, samplesCount, receiversCount * receiversCount, stream);

    bool isTextureBinded = CudaApiUtils::bindTexture1DLinear(&texSlscIqInputData, inputPtr,
                                                             samplesCount * receiversCount * receiversCount,
                                                             this->deviceProp.get());

    dim3 blockDim(1, std::min(512, height));
    dim3 gridDim(width / blockDim.x, height / blockDim.y);
    int externalSharedMemory = sizeof(float) * blockDim.x * blockDim.y * 2;
    if(isTextureBinded) {
        gpuSlscWithFocusing<true> << <
        gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount, samplesCount, receiverWidth,
            soundVelocity, samplingFrequency, startDepth, lags, 0, transmitFrequency);
        CUDA_ASSERT(cudaUnbindTexture(&texSlscIqInputData));
    } else {
        gpuSlscWithFocusing<false> << <
        gridDim, blockDim, externalSharedMemory, stream >> >(inputPtr, outputPtr, width, height, areaHeight, receiversCount, samplesCount, receiverWidth,
            soundVelocity, samplingFrequency, startDepth, lags, 0, transmitFrequency);
    }

    CUDA_ASSERT(cudaGetLastError());
}

void CudaSlscGraphNode::loadCudaDeviceProps() {
    if(!this->deviceProp) {
        this->deviceProp = boost::optional<cudaDeviceProp>(CudaApiUtils::getCudaDeviceProps());
    }
}