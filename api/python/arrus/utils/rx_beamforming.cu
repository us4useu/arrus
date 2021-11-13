#include <cupy/complex.cuh>

#define CUDART_PI_F 3.141592654f

__constant__ float xElemConst[256]; // [m]

// Assumptions:
// - TX and RX apertures have the same center position
// - x/z/angle/Elem refers to the RX aperture (i.e. the first value is the position of first aperture element, relative
// to the center of TX/RX aperture
extern "C"
__global__ void beamformPhasedArray(complex<float> *output, const complex<float> *input,
                                    float* delays, // DEBUG
                                    const unsigned nTx, const unsigned nRx, const unsigned nSamples,
                                    const float *txAngles, // [rad]
                                    const float initDelay, const float startTime,
                                    const float c, const float fs, const float fc, float maxApodTang) {
    complex<float> a, b;
    float elementX, elementZ;
    float rxTang, pixWgh = 0;
    float txDistance, rxDistance, time, s, txAngleSin, txAngleCos;
    float modSin, modCos;
    unsigned signalOffset;
    int sInt;
    unsigned sample = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned scanline = blockIdx.y * blockDim.y + threadIdx.y;
    complex<float> result = complex<float>(0.0f, 0.0f);
    complex<float> currentResult;
    complex<float> modFactor;

    if(sample >= nSamples || scanline >= nTx) {
        return;
    }

    float txAngle = txAngles[scanline];

    float r = (sample/fs + startTime)*c/2;
    __sincosf(txAngle, &txAngleSin, &txAngleCos);

    // Note: relative to the center of aperture.
    float pointX = r*txAngleSin;
    float pointZ = r*txAngleCos;
    txDistance = r;

    unsigned txOffset = scanline*nSamples*nRx;
    float cInv = 1/c;

    for(unsigned element = 0; element < nRx; ++element) {
        // Note: relative to the center of aperture.
        elementX = xElemConst[element];
        elementZ = 0; // Linear array.

        // RX apodization.
        rxTang = (pointX-elementX)/(pointZ-elementZ);
        if(fabs(rxTang) > maxApodTang) {
            continue;
        }
        // RX distance and sample number for given RX element.
        rxDistance = hypotf(elementX-pointX, elementZ-pointZ);
        time = (txDistance + rxDistance) * cInv + initDelay;
        s = time * fs;
        sInt = (int) s;

        signalOffset = txOffset + element*nSamples;
        if(sInt >= 0 && sInt < nSamples-1) {
            float ratio = s - sInt;
            a = input[signalOffset + sInt];
            b = input[signalOffset + sInt + 1];
            currentResult = (1.0f - ratio) * a + ratio * b;
        }
        else {
            continue;
        }
        __sincosf(2.0f * CUDART_PI_F*fc*time, &modSin, &modCos);
        modFactor = complex<float>(modCos, modSin);
        result = currentResult*modFactor;
        ++pixWgh;
    }
    if(pixWgh == 0.0f) {
        output[sample + scanline*nSamples] = complex<float>(0.0f, 0.0f);
    } else {
        output[sample + scanline*nSamples] = result;
    }
}