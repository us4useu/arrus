#include <cupy/complex.cuh>

#define CUDART_PI_F 3.141592654f

extern "C" __global__ void iq2RawLri(complex<float> *iqLri, const complex<float> *iqRaw,
                          const int nTx, const int nElem, const int nSamp,
                          const float *zPix, const int nZPix,
                          const float *xPix, const int nXPix,
                          const float sos, const float fs, const float fn,
                          const float *txFocuses, const float *txAngles,
                          const float pitch, const float rxApOrig,
                          const float *txApCenters, const float maxTang,
                          const float *initialDelays)
{
    // block dimensions: (tx, xSize, zSize)
    // Assumptions:
    // rx aperture -- aperture consisting of consecutive elements
    int z = blockIdx.x*blockDim.x + threadIdx.x;
    int x = blockIdx.y*blockDim.y + threadIdx.y;
    int tx = blockIdx.z*blockDim.z + threadIdx.z;

    if (z >= nZPix || x >= nXPix || tx >= nTx) {
        return;
    }
    // TODO keep in shared memory?
    float txFoc = txFocuses[tx];
    float txAng = txAngles[tx];
    float txApCent = txApCenters[tx];
    float initialDelay = initialDelays[tx];

    float txDist, rxDist, txTang, rxTang, txApod, rxApod, time, iSamp, iSampMod, interpWgh;
    float modSin, modCos, sampRe, sampIm, pixRe = 0.f, pixIm = 0.f, pixWgh = 0.f;
    complex<float> pix(0.0f, 0.0f), samp(0.0f, 0.0f), modFactor;
    float const sosInv = 1 / sos;
    float zDistInv;
    int offset;

    if (!isinf(txFoc)) {
        /* STA */
        float xFoc = txFoc * sinf(txAng) + txApCent;
        float zFoc = txFoc * cosf(txAng);

        txDist = hypotf((zPix[z] - zFoc), (xPix[x] - xFoc));
        txTang = abs((xPix[x] - xFoc))/max(abs(zPix[z] - zFoc), 1e-12);
        txApod = (fabsf(txTang) <= maxTang) ? 1.0f : 0.0f;
    }
    else {
        /* PWI */
        float lastElement = rxApOrig + (nElem-1)*pitch;
        float r1 = (xPix[x]-rxApOrig) * cosf(txAng) - zPix[z] * sinf(txAng);
        float r2 = (xPix[x]-lastElement) * cosf(txAng) - zPix[z] * sinf(txAng);

        txDist = xPix[x] * sinf(txAng) + zPix[z] * cosf(txAng);
        txApod = (r1 >= 0.f && r2 <= 0.f) ? 1.0f : 0.0f;
    }

    zDistInv = 1 / zPix[z];
    const int nSin = 512;
    float dPhase = 2 * CUDART_PI_F / (nSin-1);
    float currentPhase;
    __shared__ float2 sincosShared[nSin];
    int startSamp = threadIdx.x + threadIdx.y * blockDim.x;
    int stepSamp = blockDim.x * blockDim.y;

    for (int iSin = startSamp; iSin < nSin; iSin += stepSamp) {
        currentPhase = iSin * dPhase;
        sincosShared[iSin].x = sinf(currentPhase);
        sincosShared[iSin].y = cosf(currentPhase);
    }
    __syncthreads();

    int txOffset = tx*nSamp*nElem;

    for (int iElem = 0; iElem < nElem; iElem++) {
        float xielem = rxApOrig + (iElem)*pitch;
        rxDist = hypotf(zPix[z], (xPix[x] - xielem));
        rxTang = (xPix[x] - xielem) * zDistInv;
        rxApod = (fabsf(rxTang) <= maxTang) ? 1.0f : 0.0f;
        time = (txDist+rxDist)*sosInv + initialDelay;
        iSamp = time*fs;
        if (iSamp >= 0 && iSamp < nSamp-2) {
            offset = txOffset + iElem*nSamp;
            float originalIsamp = iSamp;
            interpWgh = modff(iSamp, &iSamp);
            int intSamp = int(iSamp);
            iSampMod = time*fn;
            iSampMod = (iSampMod - truncf(iSampMod))*(float)nSin;
            modSin = sincosShared[(int)iSampMod].x;
            modCos = sincosShared[(int)iSampMod].y;

            complex<float> modFactor = complex<float>(modCos, modSin);
            samp = iqRaw[offset+intSamp]*(1-interpWgh) + iqRaw[offset+intSamp+1]*interpWgh;
            pix += samp*modFactor*rxApod;
            pixWgh += rxApod;
        }
    }
    if(pixWgh == 0.0f) {
        iqLri[z + x*nZPix + tx*nZPix*nXPix] = complex<float>(0.0f, 0.0f);
    }
    else
        {
        iqLri[z + x*nZPix + tx*nZPix*nXPix] = pix/pixWgh*txApod;
    }
}
