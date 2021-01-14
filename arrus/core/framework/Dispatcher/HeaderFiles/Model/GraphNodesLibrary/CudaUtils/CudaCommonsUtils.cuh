#ifndef __CUDA_COMMONS_UTILS__
#define __CUDA_COMMONS_UTILS__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <math_constants.h>
#include "Model/GraphNodesLibrary/CudaUtils/CudaVectorMathUtils.cuh"

class CudaCommonsUtils {
public:
    static __device__ __forceinline__ float square(const float x) {
        return x * x;
    }

    static __device__ __forceinline__ float clamp(const float x, const float min, const float max) {
        float result = (x < min) ? min : x;
        return (result > max) ? max : result;
    }

    static __device__ __forceinline__ float
    getRealReceiver(const float receiver, const int receiversCount, const float receiverWidth) {
        return receiver / (float) (receiversCount - 1) * receiverWidth;
    }

    static __device__ __forceinline__ float
    distanceToReceiver(const float realX, const float realY, const float realReceiver) {
        return sqrtf(square(realX - realReceiver) + square(realY));
    }

    static __device__ __forceinline__ float
    distanceToReceiver(const float realX, const float realY, const int receiver, const int receiversCount,
                       const float receiverWidth) {
        float realReceiver = CudaCommonsUtils::getRealReceiver(receiver, receiversCount, receiverWidth);
        return CudaCommonsUtils::distanceToReceiver(realX, realY, realReceiver);
    }

    static __device__ __forceinline__ float
    getRealSample(const float distance, const float soundVelocity, const float samplingFrequency) {
        return distance / soundVelocity * samplingFrequency;
    }

    template<class DataType, enum cudaTextureReadMode readMode>
    static __device__ __forceinline__ DataType
    getInterpolated1DValue(const texture <DataType, cudaTextureType1D, readMode> texRef, const float realSampleIdx,
                           const int inputOffset = 0) {
        int iSampleIdx = floorf(realSampleIdx);
        float interpolationRatio = realSampleIdx - iSampleIdx;
        return (1.0f - interpolationRatio) * tex1Dfetch(texRef, inputOffset + iSampleIdx) +
               interpolationRatio * tex1Dfetch(texRef, inputOffset + iSampleIdx + 1);
    }

    template<class DataType>
    static __device__ __forceinline__ DataType
    getInterpolated1DValue(const DataType *inputData, const float realSampleIdx, const int inputOffset = 0) {
        int iSampleIdx = floorf(realSampleIdx);
        float interpolationRatio = realSampleIdx - iSampleIdx;
        return (1.0f - interpolationRatio) * inputData[inputOffset + iSampleIdx] +
               interpolationRatio * inputData[inputOffset + iSampleIdx + 1];
    }

    static __device__ __forceinline__ float
    getInterpolated2DValue(const float *data, const float x, const float y, const int width, const int height) {
        int xMin = fmax(floor(x), 0.0f);
        int xMax = fmin((float) (xMin + 1), (float) (width - 1));
        float xRest = x - (float) xMin;

        int yMin = fmax(floor(y), 0.0f);
        int yMax = fmin((float) (yMin + 1), (float) (height - 1));
        float yRest = y - (float) yMin;

        return (data[xMin + yMin * width] * (1.0f - xRest) + data[xMax + yMin * width] * xRest) * (1.0f - yRest) +
               (data[xMin + yMax * width] * (1.0f - xRest) + data[xMax + yMax * width] * xRest) * yRest;
    }

    static __device__ __forceinline__ void
    getHanningReceiversRange(const float realX, const float receiverWidth, const int receiversCount, int &startReceiver,
                             int &endReceiver, int &hanningWindowIdx) {
        float receiveElementWidth = receiverWidth / (receiversCount - 1);
        startReceiver = (realX - receiverWidth * 0.5f) / receiveElementWidth;
        endReceiver = (realX + receiverWidth * 0.5f) / receiveElementWidth;

        hanningWindowIdx = 0;
        if(startReceiver < 0) {
            hanningWindowIdx = -startReceiver;
            startReceiver = 0;
        }
        endReceiver = (endReceiver > receiversCount) ? receiversCount : endReceiver;
    }

    static __device__ __forceinline__ void
    limitReceiversByOpeningAngle(const float realX, const float realY, const float receiverWidth,
                                 const int receiversCount, int &startReceiver,
                                 int &endReceiver, int &hanningWindowIdx, const float openingAngleTangens) {
        float xOpeningAngleDistance = fabsf(realY * openingAngleTangens);
        float receiveElementWidth = receiverWidth / (receiversCount - 1);
        int startOpeningAngleReceiver = (realX - xOpeningAngleDistance) / receiveElementWidth;
        int endOpeningAngleReceiver = (realX + xOpeningAngleDistance) / receiveElementWidth;

        if(startOpeningAngleReceiver > startReceiver) {
            hanningWindowIdx += startOpeningAngleReceiver - startReceiver;
            startReceiver = startOpeningAngleReceiver;
        }

        endReceiver = (endOpeningAngleReceiver < endReceiver) ? endOpeningAngleReceiver : endReceiver;
    }

    static __device__ __forceinline__ float
    getTasinkevychConstant(const float receiverWidth, const int receiversCount, const float soundVelocity,
                           const float transmitFrequency) {
        float elementWidth = receiverWidth / receiversCount;
        float inverseWaveLength = transmitFrequency / soundVelocity;
        return CUDART_PI * elementWidth * inverseWaveLength;
    }

    static __device__ __forceinline__ float
    getTasinkevychWeight(const float realX, const float realY, const float realReceiver,
                         const float tasinkevychConstant, const float distance) {
        float currDistance = fmaxf(distance, 1e-9f);
        float thetaSin = (realReceiver - realX) / currDistance;
        float thetaCos = realY / currDistance;
        float param = tasinkevychConstant * thetaSin;
        return sinf(param) / param * thetaCos;
    }

    static __device__ __forceinline__ float degrees(const float a) {
        return 180.0f / CUDART_PI * a;
    }

    static __device__ __forceinline__ float radians(const float a) {
        return CUDART_PI / 180.0f * a;
    }

    static __device__ __forceinline__ float2
    getCurrentIQSampleValue(const float2 iqSample, const float transmitFrequency, const float totalDistance,
                            const float startDepth, const float soundVelocity) {
        float sinus, cosinus;
        __sincosf(-2.0f * CUDART_PI_F * transmitFrequency * (totalDistance - startDepth) / soundVelocity, &sinus,
                  &cosinus);
        float2 currResult;
        currResult.x = iqSample.x * cosinus - iqSample.y * sinus;
        currResult.y = iqSample.x * sinus + iqSample.y * cosinus;
        return currResult;
    }

    static __device__ __forceinline__ float getComplexModulus(const float value) {
        // Has effect only on iq data (float2). Implemented only for the templates purposes.
        return value;
    }

    static __device__ __forceinline__ float getComplexModulus(const float2 value) {
        return sqrtf(value.x * value.x + value.y * value.y + 1e-6f);
    }
};

#endif