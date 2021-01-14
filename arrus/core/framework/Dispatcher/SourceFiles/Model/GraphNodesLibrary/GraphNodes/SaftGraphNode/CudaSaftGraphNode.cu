#include "Model/GraphNodesLibrary/GraphNodes/SaftGraphNode/CudaSaftGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaTranspositionUtils.cuh"
#include "Utils/linspace.hpp"
#include "Utils/sgn.hpp"

#define _USE_MATH_DEFINES

#include <math.h>
#include <vector>
#include <algorithm>

/******************************************************************************
 * Code is not optimized
 * General TODOs:
 * 1. Adjust kernel execution limits 
 *    (from default this->allocatedSignalSize and this->allocatedApertureSize)
 * 2. Spatial C2C IFFT to C2R
 * 3. Remove evanescence part
 * 4. Hope that's all
 *
 */

/******************************************************************************
 * Constructor
 */
CudaSaftGraphNode::CudaSaftGraphNode() {
    this->inputApertureLen = 0;
    this->inputSignalLen = 0;
    this->allocatedApertureLen = 0;
    this->allocatedSignalLen = 0;
}

/******************************************************************************
 * Destructor
 */
CudaSaftGraphNode::~CudaSaftGraphNode() {
    this->releaseStructures();
}

/******************************************************************************
 * 1) Real to Complex transform
 * 2) Proper layout (3D) of data from:
 * /--------\
 * |11111122|
 * |22223333|
   |33      |
 * |        |
 * \--------/
 * into:
 * /--------\
 * |111111  |
 * |222222  |
 * |333333  |
 * |        |
 * \--------/
 */
__global__ void prepareSaftInputData(const float *const inputData,
                                     cufftComplex *const cufftData,
                                     const int apertureMaxLen,
                                     const int signalMaxLen,
                                     const int angleCount) {
    const int idxx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idxy = threadIdx.y + blockDim.y * blockIdx.y;
    const int sizex = blockDim.x * gridDim.x;
    const int sizey = blockDim.y * gridDim.y;

    cufftComplex complex;
    complex.y = 0.0f;

    const int inputBatchSize = apertureMaxLen * signalMaxLen;
    const int allignedBatchSize = sizex * sizey;

    for(int i = 0; i < angleCount; i++) {
        if(idxx < signalMaxLen && idxy < apertureMaxLen) {
            complex.x = inputData[idxx + idxy * signalMaxLen + inputBatchSize * i];
            cufftData[idxx + idxy * sizex + allignedBatchSize * i] = complex;
        } else {
            complex.x = 0.0f;
            cufftData[idxx + idxy * sizex + allignedBatchSize * i] = complex;
        }
    }
}

/******************************************************************************
 * Multiply two Complex matrices
 */
__global__ void multiplyComplexMatricesSaftFFT(cufftComplex *const inoutData,
                                               const float *const coeffsReal,
                                               const float *const coeffsImag,
                                               const int apertureMaxLen,
                                               const int signalMaxLen,
                                               const int anglesCount) {
    const int idxx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idxy = threadIdx.y + blockDim.y * blockIdx.y;
    const int sizex = blockDim.x * gridDim.x;
    const int sizey = blockDim.y * gridDim.y;

    const int allignedBatchSize = sizex * sizey;

    if(idxx < signalMaxLen && idxy < apertureMaxLen) {
        for(int i = 0; i < anglesCount; i++) {
            const int idx = idxx + idxy * sizex + allignedBatchSize * i;
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            float tmpReal = inoutData[idx].x * coeffsReal[idx] -
                            inoutData[idx].y * coeffsImag[idx];
            inoutData[idx].y = inoutData[idx].x * coeffsImag[idx] +
                               inoutData[idx].y * coeffsReal[idx];
            inoutData[idx].x = tmpReal;
        }
    }
        // Remove after tests else case, we ingore that data
    else {
        for(int i = 0; i < anglesCount; i++) {
            const int idx = idxx + idxy * sizex + allignedBatchSize * i;
            inoutData[idx].x = 0.0f;
            inoutData[idx].y = 0.0f;
        }
    }
}

/******************************************************************************
* Multiply two matrices (complex and float)
*/
__global__ void multiplyComplexFloatMatricesSaftFFT(cufftComplex *const inoutData,
                                                    const float *const coeffs,
                                                    const int apertureMaxLen,
                                                    const int signalMaxLen,
                                                    const int anglesCount) {
    const int idxx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idxy = threadIdx.y + blockDim.y * blockIdx.y;
    const int sizex = blockDim.x * gridDim.x;
    const int sizey = blockDim.y * gridDim.y;

    const int allignedBatchSize = sizex * sizey;

    if(idxx < signalMaxLen && idxy < apertureMaxLen) {
        for(int i = 0; i < anglesCount; i++) {
            const int idx = idxx + idxy * sizex + allignedBatchSize * i;
            inoutData[idx].x *= coeffs[idx];
            inoutData[idx].y *= coeffs[idx];
        }
    }
}

/******************************************************************************
 * 
 */
__global__ void conjAndShiftSaftFFT(cufftComplex *const cufftData,
                                    const int angleCount) {
    int idxx = blockIdx.x * blockDim.x + threadIdx.x;
    int idxy = blockIdx.y * blockDim.y + threadIdx.y;
    int sizex = gridDim.x * blockDim.x;
    int sizey = gridDim.y * blockDim.y;

    if(idxx >= sizex / 2 + 1) {
        for(int angle = 0; angle < angleCount; angle++) {
            int idxAngleOffset = angle * sizex * sizey;
            int idxxm;
            int idxym;

            if(idxy == 0) {
                idxxm = sizex - idxx;
                idxym = 0;
            } else {
                idxxm = sizex - idxx;
                idxym = sizey - idxy;
            }

            int indexWrite = idxx + idxy * sizex + idxAngleOffset;
            int indexRead = idxxm + idxym * sizex + idxAngleOffset;
            cufftData[indexWrite].x = +cufftData[indexRead].x;
            cufftData[indexWrite].y = -cufftData[indexRead].y;
        }
    }
}

/******************************************************************************
 * Interpolation kernel
 */
__global__ void interpLinSaftFFT(const cufftComplex *const inData,
                                 cufftComplex *const outData,
                                 const float *const fkz,
                                 const int angleCount,
                                 const int shotSize) {
    const int idxx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idxy = threadIdx.y + blockDim.y * blockIdx.y;
    const int sizex = blockDim.x * gridDim.x;

    for(int angle = 0; angle < angleCount; angle++) {
        const int idx = idxx + idxy * sizex + angle * shotSize;
        const int idxOffset = idxy * sizex + angle * shotSize;
        // If from [0; signalSize / 2 + 1]
        if(fkz[idx] <= sizex / 2 && idxx <= sizex / 2 + 1) {
            int idxf = (const int) floorf(fkz[idx]);
            const float idxk = (float) idxf - fkz[idx];
            idxf += idxOffset;
            outData[idx].x = inData[idxf].x * (idxk + 1.0f) - (inData[idxf + 1].x) * idxk;
            outData[idx].y = inData[idxf].y * (idxk + 1.0f) - (inData[idxf + 1].y) * idxk;
        } else {
            outData[idx] = {0.0f, 0.0f};
        }
    }

}

/******************************************************************************
 * Merges data from multiple angles results
 */
__global__ void mergeSaftFFT(const cufftComplex *const inputData,
                             cufftComplex *const outputData,
                             const int apertureMaxLen,
                             const int signalMaxLen,
                             const int angleCount,
                             const int shotSize) {
    int idxx = threadIdx.x + blockDim.x * blockIdx.x;
    int idxy = threadIdx.y + blockDim.y * blockIdx.y;
    int sizex = blockDim.x * gridDim.x;

    cufftComplex result = {0.0f, 0.0f};

    if(idxx < signalMaxLen && idxy < apertureMaxLen) {
        for(int angle = 0; angle < angleCount; angle++) {
            result.x += inputData[idxx + idxy * sizex + shotSize * angle].x;
            result.y += inputData[idxx + idxy * sizex + shotSize * angle].y;
        }
    }

    result.x /= angleCount;
    result.y /= angleCount;

    outputData[idxx + idxy * sizex] = result;
}

/******************************************************************************
 * 1) Complex to Real (Abs of real and imag)
 * 2) Proper data layout
 */
__global__ void absSaftFFT(const cufftComplex *const cufftData,
                           float *const outputData,
                           const int apertureLen,
                           const int signalLen) {
    int idxx = threadIdx.x + blockDim.x * blockIdx.x;
    int idxy = threadIdx.y + blockDim.y * blockIdx.y;
    int sizex = blockDim.x * gridDim.x;

    if(idxx < signalLen && idxy < apertureLen) {
        float result;
        cufftComplex complex = cufftData[idxx + idxy * sizex];
        result = sqrtf(complex.x * complex.x + complex.y * complex.y);
        outputData[idxx + idxy * signalLen] = powf(result, 0.7f);
    }
}

/******************************************************************************
 * 
 */
std::vector<float> CudaSaftGraphNode::generateKx(const float pitch) {
    std::vector<float> kx = linspace(0.0f,
                                     1.0f / pitch / 2.0f,
                                     this->allocatedApertureLen / 2 + 1);

    std::vector<float> kx2 = linspace(-kx[this->allocatedApertureLen / 2 - 2],
                                      -kx[1],
                                      this->allocatedApertureLen / 2 - 1);

    kx.insert(kx.end(), kx2.begin(), kx2.end());
    kx2.clear();

    return kx;
}

/******************************************************************************
 * 
 */
std::vector<float> CudaSaftGraphNode::generateF0(const float fs) {
    std::vector<float> f0 = linspace(0.0f, fs / 2, this->allocatedSignalLen / 2 + 1);

    return f0;
}

/******************************************************************************
 * Generates FKZ matrix for every angle shot
 */
std::vector<float> CudaSaftGraphNode::generateFkz(const std::vector<float> f0,
                                                  const std::vector<float> kx,
                                                  const float soundVelocity,
                                                  const float fs,
                                                  const float pitch) {
    int dataElements = this->allocatedSignalLen * this->allocatedApertureLen * (const int) this->allocatedAngles.size();

    std::vector<float> fkz = std::vector<float>();
    fkz.reserve(dataElements);

    // For every angle
    for(int angle = 0; angle < (const int) this->allocatedAngles.size(); angle++) {
        // Degrees to radians
        const float angleRad = (float) (this->allocatedAngles[angle] * M_PI / 180.0f);
        float sinA = sinf(angleRad);
        float cosA = cosf(angleRad);

        // ERM velocity
        float ERMv = soundVelocity / sqrtf(1.0f + cosA + sinA * sinA);

        // Note: we choose kz = 2 * f / c(i.e.z = c*t / 2);
        float C = (1.0f + cosA + sinA * sinA) / pow(1.0f + cosA, 1.5f);

        for(int a = 0; a < kx.size(); a++) // Aperture
        {
            for(int t = 0; t < this->allocatedSignalLen; t++) // Time
            {
                if(t < f0.size()) {
                    fkz.push_back(ERMv * sqrtf(kx[a] * kx[a] + 4.0f * f0[t] * f0[t] /
                                                               (soundVelocity * soundVelocity) * (C * C)) /
                                  (fs / (float) this->allocatedSignalLen));
                } else {
                    fkz.push_back(0.0f);
                }
            }
        }
    }

    return fkz;
}

/******************************************************************************
 * Generates matrix for compensation steering angle and/or 
 * depth start for every angle shot
 */
std::vector<float> CudaSaftGraphNode::generateForwardCoefficients(const float soundVelocity,
                                                                  const float fs,
                                                                  const float pitch,
                                                                  const float t0) {
    int dataElements = this->allocatedSignalLen * this->allocatedApertureLen * (const int) this->allocatedAngles.size();

    std::vector<float> real = std::vector<float>();
    std::vector<float> imag = std::vector<float>();
    real.reserve(dataElements);
    imag.reserve(dataElements);

    // 2n + 1
    std::vector<float> f0 = linspace(0.0f, fs / 2, this->allocatedSignalLen / 2 + 1);

    // For every angle
    for(int angle = 0; angle < (const int) this->allocatedAngles.size(); angle++) {
        // Degrees to radians
        const float angleRad = (float) (this->allocatedAngles[angle] * M_PI / 180.0f);
        float sinA = sinf(angleRad);
        float cosA = cosf(angleRad);

        // ERM velocity
        float ERMv = soundVelocity / sqrt(1.0f + cosA + sinA * sinA);

        // Compensate for steering angle and/or depth start
        if(sinA != 0.0f || t0 != 0.0f) {
            for(int el = 0; el < this->allocatedApertureLen; el++) {
                for(int t = 0; t < this->allocatedSignalLen; t++) {
                    if(t < f0.size() && el < this->inputApertureLen) {
                        float dt = sinA * ((this->inputApertureLen - 1) * (angleRad < 0) - el) * pitch / soundVelocity;
                        dt = f0[t] * (dt + t0);
                        real.push_back(cosf(-2.0f * (float) M_PI * dt));
                        imag.push_back(sinf(-2.0f * (float) M_PI * dt));
                    } else {
                        real.push_back(0.0f);
                        imag.push_back(0.0f);
                    }
                }
            }
        } else {
            for(int el = 0; el < this->allocatedApertureLen; el++) {
                for(int t = 0; t < this->allocatedSignalLen; t++) {
                    if(t < f0.size()) {
                        real.push_back(1.0f);
                        imag.push_back(0.0f);
                    } else {
                        real.push_back(0.0f);
                        imag.push_back(0.0f);
                    }
                }
            }
        }
    }

    // Pack two vectors
    real.insert(real.end(), imag.begin(), imag.end());
    return real;
}

/******************************************************************************
 * Generates matrix for compensation steering angle for every angle shot 
 * (inverse)
 */
std::vector<float> CudaSaftGraphNode::generateInverseCoefficients(const std::vector<float> kx,
                                                                  const float soundVelocity,
                                                                  const float fs,
                                                                  const float pitch) {
    int dataElements = this->allocatedSignalLen * this->allocatedApertureLen * (const int) this->allocatedAngles.size();

    std::vector<float> real = std::vector<float>();
    std::vector<float> imag = std::vector<float>();
    real.reserve(dataElements);
    imag.reserve(dataElements);

    // For every angle
    for(int angle = 0; angle < (const int) this->allocatedAngles.size(); angle++) {
        // Degrees to radians
        const float angleRad = (float) (this->allocatedAngles[angle] * M_PI / 180.0f);
        float sinA = sinf(angleRad);
        float cosA = cosf(angleRad);

        float C = sinA / (2.0f - cosA);

        // Compensate for steering angle
        if(sinA != 0.0f) {
            for(int el = 0; el < this->allocatedApertureLen; el++) {
                for(int t = 0; t < this->allocatedSignalLen; t++) {
                    float dx = -C * t / fs * soundVelocity / 2;
                    dx = kx[el] * dx;

                    // Division for normalization of CUDA's IFFT
                    real.push_back(cosf(-2.0f * (float) M_PI * dx) / this->allocatedSignalLen);
                    imag.push_back(sinf(-2.0f * (float) M_PI * dx) / this->allocatedSignalLen);
                }
            }
        } else {
            for(int el = 0; el < this->allocatedApertureLen; el++) {
                for(int t = 0; t < this->allocatedSignalLen; t++) {
                    // Division for normalization of CUDA's IFFT
                    real.push_back(1.0f / this->allocatedSignalLen);
                    imag.push_back(0.0f);
                }
            }
        }
    }

    // Pack two vectors
    real.insert(real.end(), imag.begin(), imag.end());
    return real;
}

std::vector<float> CudaSaftGraphNode::generateObliquityFactor(const std::vector<float> f0,
                                                              const std::vector<float> fkz,
                                                              const float fs) {
    std::vector<float> res = std::vector<float>();
    res.reserve(fkz.size());

    // For every angle
    for(int angle = 0; angle < (const int) this->allocatedAngles.size(); angle++) {
        for(int el = 0; el < this->allocatedApertureLen; el++) {
            for(int t = 0; t < this->allocatedSignalLen; t++) {
                float divider = fkz[t + this->allocatedSignalLen * el +
                                    this->allocatedApertureLen * this->allocatedSignalLen * angle];

                if(divider != 0.0f)
                    res.push_back(f0[t] / divider * (fs / this->allocatedSignalLen));
                else
                    res.push_back(0.0f);
            }
        }
    }

    // First value is NaN
    res[0] = 0.0f;

    return res;
}

/******************************************************************************
 * Allocs structures for
 */
void CudaSaftGraphNode::allocStructures(const int apertureLen,
                                        const int signalLen,
                                        const std::vector<float> anglesInfo,
                                        const float t0,
                                        const float soundVelocity,
                                        const float fs,
                                        const float pitch) {
    // TODO: Verify difference in angles
    if(apertureLen != this->inputApertureLen ||
       signalLen != this->inputSignalLen) {
        this->releaseStructures();

        // Rememeber what allocated
        this->inputApertureLen = apertureLen;
        this->inputSignalLen = signalLen;
        this->allocatedAngles = anglesInfo;
        // Next powers of 2
        // Multiply by two: extensive 0-padding is required with linear interpolation
        this->allocatedSignalLen = (int) pow(2.0f, (int) ceilf(log2((float) signalLen))) * 2;
        // Mulitply by two in order to avoid lateral edge effects
        this->allocatedApertureLen = (int) pow(2.0f, (int) ceilf(log2((float) apertureLen))) * 2;

        // Helper variable
        const int dataElements =
            this->allocatedSignalLen * this->allocatedApertureLen * (const int) this->allocatedAngles.size();

        // Alloc memory for Compensation coefficients on CPU
        std::vector<float> kx = generateKx(pitch);
        std::vector<float> f0 = generateF0(fs);
        std::vector<float> fkz = generateFkz(f0, kx, soundVelocity, fs, pitch);
        std::vector<float> forComplexCoeffs = generateForwardCoefficients(soundVelocity, fs, pitch, t0);
        std::vector<float> invComplexCoeffs = generateInverseCoefficients(kx, soundVelocity, fs, pitch);
        std::vector<float> obliquityFactor = generateObliquityFactor(f0, fkz, fs);
        f0.clear();
        kx.clear();

        // Split real and imaginary parts
        const int halfForSize = (const int) forComplexCoeffs.size() / 2;
        const int halfInvSize = (const int) invComplexCoeffs.size() / 2;
        std::vector<float> forRealCoeffs(forComplexCoeffs.begin(), forComplexCoeffs.begin() + halfForSize);
        std::vector<float> forImagCoeffs(forComplexCoeffs.begin() + halfForSize, forComplexCoeffs.end());
        std::vector<float> invRealCoeffs(invComplexCoeffs.begin(), invComplexCoeffs.begin() + halfInvSize);
        std::vector<float> invImagCoeffs(invComplexCoeffs.begin() + halfInvSize, invComplexCoeffs.end());
        forComplexCoeffs.clear();
        invComplexCoeffs.clear();

        // Alloc memory for Compensation coefficients on GPU
        CUDA_ASSERT(cudaMalloc((void **) &this->devFkz,
                               sizeof(float) * dataElements));
        CUDA_ASSERT(cudaMalloc((void **) &this->devForwardRealCompensation,
                               sizeof(float) * dataElements));
        CUDA_ASSERT(cudaMalloc((void **) &this->devForwardImagCompensation,
                               sizeof(float) * dataElements));
        CUDA_ASSERT(cudaMalloc((void **) &this->devInverseRealCompensation,
                               sizeof(float) * dataElements));
        CUDA_ASSERT(cudaMalloc((void **) &this->devInverseImagCompensation,
                               sizeof(float) * dataElements));
        CUDA_ASSERT(cudaMalloc((void **) &this->devObliquityFactor,
                               sizeof(float) * dataElements));

        // CuFFT Library allocs
        CUDA_ASSERT(cudaMalloc((void **) &this->devCufftData,
                               sizeof(cufftComplex) * dataElements));
        CUDA_ASSERT(cudaMalloc((void **) &this->devCufftOutData,
                               sizeof(cufftComplex) * dataElements));

        CUDA_ASSERT(cudaMalloc((void **) &this->devOutputBuffer,
                               sizeof(float) * this->inputApertureLen * this->inputSignalLen));

        // Batched 1D FFT
        CUFFT_ASSERT(cufftPlan1d(&this->temporalPlan,
                                 this->allocatedSignalLen,
                                 CUFFT_C2C,
                                 this->allocatedApertureLen * (const int) this->allocatedAngles.size()));

        int dims[1] = {this->allocatedApertureLen};
        // For only one angle and for half of frequencies (n/2+1) only
        CUFFT_ASSERT(cufftPlanMany(&this->spatialPlan, 1, dims,
                                   dims, this->allocatedSignalLen, 1,
                                   dims, this->allocatedSignalLen, 1,
                                   CUFFT_C2C, this->allocatedSignalLen / 2 + 1));

        // Copy Compensation coefficients CPU -> GPU
        CUDA_ASSERT(cudaMemcpy(this->devFkz,
                               (void *) &fkz[0],
                               sizeof(float) * dataElements,
                               cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(this->devForwardRealCompensation,
                               (void *) &forRealCoeffs[0],
                               sizeof(float) * dataElements,
                               cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(this->devForwardImagCompensation,
                               (void *) &forImagCoeffs[0],
                               sizeof(float) * dataElements,
                               cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(this->devInverseRealCompensation,
                               (void *) &invRealCoeffs[0],
                               sizeof(float) * dataElements,
                               cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(this->devInverseImagCompensation,
                               (void *) &invImagCoeffs[0],
                               sizeof(float) * dataElements,
                               cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(this->devObliquityFactor,
                               (void *) &obliquityFactor[0],
                               sizeof(float) * dataElements,
                               cudaMemcpyHostToDevice));

        // Clear CPU Memory
        fkz.clear();
        forRealCoeffs.clear();
        forImagCoeffs.clear();
        invRealCoeffs.clear();
        invImagCoeffs.clear();
        obliquityFactor.clear();
    }
}

/******************************************************************************
 * Deallocs previously allocated structures
 */
void CudaSaftGraphNode::releaseStructures() {
    if(this->inputApertureLen != 0 &&
       this->inputSignalLen != 0 &&
       !this->allocatedAngles.empty()) {
        CUDA_ASSERT(cudaFree(this->devFkz));
        CUDA_ASSERT(cudaFree(this->devForwardRealCompensation));
        CUDA_ASSERT(cudaFree(this->devForwardImagCompensation));
        CUDA_ASSERT(cudaFree(this->devInverseRealCompensation));
        CUDA_ASSERT(cudaFree(this->devInverseImagCompensation));
        CUDA_ASSERT(cudaFree(this->devObliquityFactor));
        CUDA_ASSERT(cudaFree(this->devCufftData));
        CUDA_ASSERT(cudaFree(this->devCufftOutData));
        CUDA_ASSERT(cudaFree(this->devOutputBuffer));
        CUFFT_ASSERT(cufftDestroy(this->temporalPlan));
        CUFFT_ASSERT(cufftDestroy(this->spatialPlan));
    }
}

/******************************************************************************
 * Entry point for SAFT-FFT
 */
void CudaSaftGraphNode::saft(const float *const inputPtr,
                             float *const outputPtr,
                             const int apertureLen,
                             const int signalLen,
                             const std::vector<float> anglesInfo,
                             const float t0,
                             const float soundVelocity,
                             const float fs,
                             const float pitch,
                             const cudaStream_t &stream) {
    this->allocStructures(apertureLen, signalLen, anglesInfo, t0, soundVelocity, fs, pitch);

    int shotSize = this->allocatedApertureLen * this->allocatedSignalLen;

    dim3 block(16, 16);
    dim3 grid(this->allocatedSignalLen / block.x, this->allocatedApertureLen / block.y);

    // Prepare whole block layout data from compact real float to aligned cufftComplex
    prepareSaftInputData << < grid, block, 0, stream >> >(inputPtr,
        this->devCufftData,
        this->inputApertureLen,
        this->inputSignalLen,
        (const int) anglesInfo.size());
    CUDA_ASSERT(cudaGetLastError());

    // Set stream for FFT Plans
    CUFFT_ASSERT(cufftSetStream(this->temporalPlan, stream));
    CUFFT_ASSERT(cufftSetStream(this->spatialPlan, stream));

    // Execute temporal FFT
    CUFFT_ASSERT(cufftExecC2C(this->temporalPlan,
                              this->devCufftData,
                              this->devCufftData,
                              CUFFT_FORWARD));

    // Note that we changes operation sizes during algorithm

    // Mulitply only user data (time domain)
    // Compensation for steering angle and/or depth start
    multiplyComplexMatricesSaftFFT << < grid, block, 0, stream >> >(this->devCufftData,
        this->devForwardRealCompensation,
        this->devForwardImagCompensation,
        this->inputApertureLen,
        this->allocatedSignalLen / 2 + 1,
        (const int) anglesInfo.size());
    CUDA_ASSERT(cudaGetLastError());

    for(int angle = 0; angle < (const int) anglesInfo.size(); angle++) {
        // Column-major FFT
        CUFFT_ASSERT(cufftExecC2C(this->spatialPlan,
                                  this->devCufftData + angle * this->allocatedApertureLen * this->allocatedSignalLen,
                                  this->devCufftData + angle * this->allocatedApertureLen * this->allocatedSignalLen,
                                  CUFFT_FORWARD));
    }

    // TODO Remove evanescent parts
    //%isevanescent = abs(f) . / abs(kx) < c;
    //%SIGk(isevanescent) = 0;

    // Interpolation in the frequency domain: f -> fkz
    interpLinSaftFFT << < grid, block, 0, stream >> >(this->devCufftData,
        this->devCufftOutData,
        this->devFkz,
        (const int) this->allocatedAngles.size(),
        shotSize);
    CUDA_ASSERT(cudaGetLastError());

    // Obliquity factor
    multiplyComplexFloatMatricesSaftFFT << < grid, block, 0, stream >> >(this->devCufftOutData,
        this->devObliquityFactor,
        this->allocatedApertureLen,
        this->allocatedSignalLen / 2 + 1,
        (const int) anglesInfo.size());
    CUDA_ASSERT(cudaGetLastError());


    conjAndShiftSaftFFT << < grid, block, 0, stream >> >(this->devCufftOutData,
        (const int) this->allocatedAngles.size());
    auto er = cudaGetLastError();
    CUDA_ASSERT(er);

    CUFFT_ASSERT(cufftExecC2C(this->temporalPlan,
                              this->devCufftOutData,
                              this->devCufftOutData,
                              CUFFT_INVERSE));

    // Mulitply all data (frequency domain)
    // Compensation for steering angle
    // Including data normalization inside inverseCompensation
    multiplyComplexMatricesSaftFFT << < grid, block, 0, stream >> >(this->devCufftOutData,
        this->devInverseRealCompensation,
        this->devInverseImagCompensation,
        this->allocatedApertureLen,
        this->allocatedSignalLen,
        (const int) anglesInfo.size());
    CUDA_ASSERT(cudaGetLastError());

    // Compounding
    mergeSaftFFT << < grid, block, 0, stream >> >(this->devCufftOutData,
        this->devCufftOutData,
        this->allocatedApertureLen,
        this->allocatedSignalLen,
        (const int) anglesInfo.size(),
        shotSize);
    CUDA_ASSERT(cudaGetLastError());

    // For only first shot which is compounded
    CUFFT_ASSERT(cufftExecC2C(this->spatialPlan,
                              this->devCufftOutData,
                              this->devCufftOutData,
                              CUFFT_INVERSE));

    // Merge and compute absolute value
    // Note that no normalization after spatial C2C IFFT
    absSaftFFT << < grid, block, 0, stream >> >(this->devCufftOutData,
        this->devOutputBuffer,
        this->inputApertureLen,
        this->inputSignalLen);
    CUDA_ASSERT(cudaGetLastError());

    CudaTranspositionUtils::transpose(this->devOutputBuffer, outputPtr, this->inputSignalLen, this->inputApertureLen,
                                      stream);
}
