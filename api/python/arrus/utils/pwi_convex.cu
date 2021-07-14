#include <cupy/complex.cuh>

#define CUDART_PI_F 3.141592654f

// TODO usunac tekstury DONE
// TODO przygotowac parametry wywolania

extern "C"
__global__ void
iqRaw2LriConvex(complex<float> *iqLri, const complex<float> *iqRaw,
                const float *xElem, const float *zElem, const float *tangElem, const int nElem,
                const int nTx, const int nSamp,
                const float *zPix, const int nZPix,
                const float *xPix, const int nXPix,
                float const sos, float const fs, float const fn,
                const float *txAngZX, const float *txApCentZ, const float *txApCentX,
                // TODO below? both inclusive? element number
                const int *txApFstElem, const int *txApLstElem,
                const int *rxApOrigElem, const int nRx,
                const float minRxTang, const float maxRxTang,
                float const initDel) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (z >= nZPix || x >= nXPix) {
        return;
    }

    int iElem, offset;
    float interpWgh;
    float txDist, rxDist, rxTang, txApod, rxApod, time, iSamp;
    float modSin, modCos, sampRe, sampIm, pixRe, pixIm, pixWgh;
    const float omega = 2 * CUDART_PI_F * fn;
    const float sosInv = 1 / sos;
    const float nSigma = 3; // number of sigmas in half of the apodization Gaussian curve
    const float twoSigSqrInv = nSigma * nSigma * 0.5f;
    const float rngRxTangInv = 2 / (maxRxTang - minRxTang); // inverted half range
    const float centRxTang = (maxRxTang + minRxTang) * 0.5f;
    complex<float> pix(0.0f, 0.0f), samp(0.0f, 0.0f), modFactor;

    for (int iTx = 0; iTx < nTx; ++iTx) {
        int txOffset = iTx*nSamp*nRx;
        /* PWI */
        txDist = (zPix[z] - txApCentZ[iTx]) * cosf(txAngZX[iTx]) +
                 (xPix[x] - txApCentX[iTx]) * sinf(txAngZX[iTx]);

        // Projections of ApEdge-Pix vector on the rotated unit vector of tx direction (dot products) ...
        // to determine if the pixel is in the sonified area (dot product >= 0).
        // For ApEdgeFst, the vector is rotated left, for ApEdgeLst the vector is rotated right.
        txApod = (((-(zPix[z] - zElem[txApFstElem[iTx]]) *
                    sinf(txAngZX[iTx]) +
                    (xPix[x] - xElem[txApFstElem[iTx]]) *
                    cosf(txAngZX[iTx])) >= 0.f) &&
                  (((zPix[z] - zElem[txApLstElem[iTx]]) *
                    sinf(txAngZX[iTx]) -
                    (xPix[x] - xElem[txApLstElem[iTx]]) *
                    cosf(txAngZX[iTx])) >= 0.f)) ? 1.f : 0.f;

        pixWgh = 0.f;

        if (txApod != 0.f) {
            for (int iRx = 0; iRx < nRx; iRx++) {
                iElem = iRx + rxApOrigElem[iTx];
                if (iElem < 0 || iElem >= nElem) continue;

                rxDist = hypotf(xPix[x] - xElem[iElem], zPix[z] - zElem[iElem]);
                rxTang = __fdividef(xPix[x] - xElem[iElem], zPix[z] - zElem[iElem]);
                rxTang = __fdividef(rxTang - tangElem[iElem], 1.f + rxTang*tangElem[iElem]);
                if (rxTang < minRxTang || rxTang > maxRxTang) continue;

                rxApod = (rxTang - centRxTang) * rngRxTangInv;
                rxApod = __expf(-rxApod * rxApod * twoSigSqrInv);

                time = (txDist + rxDist)*sosInv + initDel;
                iSamp = time * fs;
                if (iSamp < 0.f || iSamp >= static_cast<float>(nSamp - 1)) {
                    continue;
                }
                offset = txOffset + iRx*nSamp;
                interpWgh = modff(iSamp, &iSamp);
                int intSamp = int(iSamp);

                __sincosf(omega*time, &modSin, &modCos);
                complex<float> modFactor = complex<float>(modCos, modSin);

                samp = iqRaw[offset+intSamp]*(1-interpWgh) + iqRaw[offset+intSamp+1]*interpWgh;
                pix += samp*modFactor*rxApod;
                pixWgh += rxApod;
            }
        }
        if(pixWgh == 0.0f) {
            iqLri[z + x*nZPix + iTx*nZPix*nXPix] = complex<float>(0.0f, 0.0f);
        }
        else {
            iqLri[z + x * nZPix + iTx * nZPix * nXPix] = pix/pixWgh*txApod;
        }
    }

}