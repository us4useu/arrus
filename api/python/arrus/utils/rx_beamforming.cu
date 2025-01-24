#include <cupy/complex.cuh>

#define CUDART_PI_F 3.141592654f

// The below values are stored for ALL probe elements.
__constant__ float xElemConst[256]; // [m]
__constant__ float zElemConst[256]; // [m]
__constant__ float angleElemConst[256]; // [rad]

// Assumptions:
// - TX and RX apertures have the same center position
extern "C"
    __global__ void beamform(complex<float> *output, const complex<float> *input,
             const unsigned nSeq, const unsigned nTx, const unsigned nRx, const unsigned nSamples,
             const float *txAngles, // [rad]
             const float initDelay, const float startTime,
             const float c, const float fs, const float fc, float maxApodTang,
             const size_t xElemConstOffset, const size_t zElemConstOffset, const size_t angleElemConstOffset) {
    complex<float> a, b;
    float elementX, elementZ, elementAngle;
    float rxAng, rxTang, pixWgh = 0;
    float txDistance, rxDistance, time, s, txAngleSin, txAngleCos;
    float modSin, modCos;
    unsigned signalOffset;
    int sInt;
    complex<float> result = complex<float>(0.0f, 0.0f);
    complex<float> currentResult;
    complex<float> modFactor;

    unsigned sample = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned scanline = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned frame = blockIdx.z * blockDim.z + threadIdx.z;

    if(sample >= nSamples || scanline >= nTx || frame >= nSeq) {
        return;
    }
    float txAngle = txAngles[scanline];

    float r = (sample/fs + startTime)*c/2;
    __sincosf(txAngle, &txAngleSin, &txAngleCos);

    // Note: relative to the center of aperture.
    // coordinate system: aperture's center
    float pointX = r*txAngleSin;
    float pointZ = r*txAngleCos;
    txDistance = r;

    unsigned txOffset = frame*nTx*nRx*nSamples + scanline*nRx*nSamples;
    float cInv = 1/c;

    for(int element = 0; element < nRx; ++element) {
        elementX = xElemConst[xElemConstOffset + element];
        elementZ = zElemConst[zElemConstOffset + element];
        elementAngle = angleElemConst[angleElemConstOffset + element];

        // RX apodization.
        rxAng = atan2f(pointX-elementX, pointZ-elementZ);
        rxAng = rxAng-elementAngle;
        rxTang = tanf(rxAng);

        if(fabs(rxTang) > maxApodTang) {
            continue;
        }
        // RX distance and sample number for a given RX element.
        rxDistance = hypotf(elementX-pointX, elementZ-pointZ);
        time = (txDistance+rxDistance)*cInv + initDelay;
        s = time*fs;
        sInt = (int) s;

        signalOffset = txOffset + element*nSamples;
        if(sInt >= 0 && sInt < nSamples-1) {
            float ratio = s - sInt;
            a = input[signalOffset + sInt];
            b = input[signalOffset + sInt + 1];
            currentResult = (1.0f - ratio)*a + ratio*b;
        }
        else if(sInt == nSamples-1) {
            currentResult = input[signalOffset + sInt];
        }
        else {
            continue;
        }
        __sincosf(2.0f*CUDART_PI_F*fc*time, &modSin, &modCos);
        modFactor = complex<float>(modCos, modSin);
        result += currentResult*modFactor;
        ++pixWgh;
    }
    if(pixWgh == 0.0f) {
        output[frame*nTx*nSamples + scanline*nSamples + sample] = complex<float>(0.0f, 0.0f);
    } else {
        output[frame*nTx*nSamples + scanline*nSamples + sample] = result/pixWgh;
    }
}