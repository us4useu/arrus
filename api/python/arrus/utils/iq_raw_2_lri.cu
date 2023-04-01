#include <cupy/complex.cuh>

#define CUDART_PI_F 3.141592654f

__constant__ float zElemConst[256];
__constant__ float xElemConst[256];
__constant__ float tangElemConst[256];

extern "C"
__global__ void
iqRaw2Lri(complex<float> *iqLri, const complex<float> *iqRaw,
          const int nElem,
          const int nSeq, const int nTx, const int nSamp,
          const float *zPix, const int nZPix,
          const float *xPix, const int nXPix,
          float const sos, float const fs, float const fn,
          const float *txFoc, const float *txAngZX,
          const float *txApCentZ, const float *txApCentX,
          const int *txApFstElem, const int *txApLstElem,
          const int *rxApOrigElem, const int nRx,
          const float minRxTang, const float maxRxTang,
          float const initDel) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int iGlobalTx = blockIdx.z * blockDim.z + threadIdx.z;

    if(z >= nZPix || x >= nXPix || iGlobalTx >= nSeq*nTx) {
        return;
    }
    int iTx = iGlobalTx % nTx;

    int iElem, offset;
    float interpWgh;
    float txDist, rxDist, rxTang, txApod, rxApod, time, iSamp;
    float modSin, modCos, pixWgh;
    const float omega = 2 * CUDART_PI_F * fn;
    const float sosInv = 1 / sos;
    const float nSigma = 3; // number of sigmas in half of the apodization Gaussian curve
    const float twoSigSqrInv = nSigma * nSigma * 0.5f;
    const float rngRxTangInv = 2 / (maxRxTang - minRxTang); // inverted half range
    const float centRxTang = (maxRxTang + minRxTang) * 0.5f;
    complex<float> pix(0.0f, 0.0f), samp(0.0f, 0.0f), modFactor;

    int txOffset = iGlobalTx * nSamp * nRx;

    if(!isinf(txFoc[iTx])) {
        /* STA */
        float zFoc = txApCentZ[iTx] + txFoc[iTx] * cosf(txAngZX[iTx]);
        float xFoc = txApCentX[iTx] + txFoc[iTx] * sinf(txAngZX[iTx]);

        float pixFocArrang;

        if(txFoc[iTx] <= 0.0f) {
            /* Virtual Point Source BEHIND probe surface */
            // Valid pixels are assumed to be always in front of the focal point (VSP)
            pixFocArrang = 1.0f;
        } else {
            /* Virtual Point Source IN FRONT OF probe surface */
            // Projection of the Foc-Pix vector on the ApCent-Foc vector (dot product) ...
            // to determine if the pixel is behind (-) or in front of (+) the focal point (VSP).
            pixFocArrang = (((zPix[z] - zFoc) * (zFoc - txApCentZ[iTx]) +
                             (xPix[x] - xFoc) * (xFoc - txApCentX[iTx])) >= 0.f) ? 1.f : -1.f;
        }
        txDist = hypotf(zPix[z] - zFoc, xPix[x] - xFoc);
        txDist *= pixFocArrang; // Compensation for the Pix-Foc arrangement
        txDist += txFoc[iTx]; // Compensation for the reference time being the moment when txApCent fires.

        // Projections of Foc-Pix vector on the rotated Foc-ApEdge vectors (dot products) ...
        // to determine if the pixel is in the sonified area (dot product >= 0).
        // Foc-ApEdgeFst vector is rotated left, Foc-ApEdgeLst vector is rotated right.
        txApod = (((-(xElemConst[txApFstElem[iTx]] - xFoc) * (zPix[z] - zFoc) +
                    (zElemConst[txApFstElem[iTx]] - zFoc) * (xPix[x] - xFoc)) * pixFocArrang >= 0.f) &&
                  (((xElemConst[txApLstElem[iTx]] - xFoc) * (zPix[z] - zFoc) -
                    (zElemConst[txApLstElem[iTx]] - zFoc) * (xPix[x] - xFoc)) * pixFocArrang >= 0.f)) ? 1.f : 0.f;
    } else {
        /* PWI */
        txDist = (zPix[z] - txApCentZ[iTx]) * cosf(txAngZX[iTx]) +
                 (xPix[x] - txApCentX[iTx]) * sinf(txAngZX[iTx]);

        // Projections of ApEdge-Pix vector on the rotated unit vector of tx direction (dot products) ...
        // to determine if the pixel is in the sonified area (dot product >= 0).
        // For ApEdgeFst, the vector is rotated left, for ApEdgeLst the vector is rotated right.
        txApod = (((-(zPix[z] - zElemConst[txApFstElem[iTx]]) * sinf(txAngZX[iTx]) +
                    (xPix[x] - xElemConst[txApFstElem[iTx]]) * cosf(txAngZX[iTx])) >= 0.f) &&
                  (((zPix[z] - zElemConst[txApLstElem[iTx]]) * sinf(txAngZX[iTx]) -
                    (xPix[x] - xElemConst[txApLstElem[iTx]]) * cosf(txAngZX[iTx])) >= 0.f)) ? 1.f : 0.f;
    }
    pixWgh = 0.0f;
    pix.real(0.0f);
    pix.imag(0.0f);

    if(txApod != 0.0f) {
        for(int iRx = 0; iRx < nRx; iRx++) {
            iElem = iRx + rxApOrigElem[iTx];
            if(iElem < 0 || iElem >= nElem) continue;

            rxDist = hypotf(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
            rxTang = __fdividef(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
            rxTang = __fdividef(rxTang - tangElemConst[iElem], 1.f + rxTang * tangElemConst[iElem]);
            if(rxTang < minRxTang || rxTang > maxRxTang) continue;

            rxApod = (rxTang - centRxTang) * rngRxTangInv;
            rxApod = __expf(-rxApod * rxApod * twoSigSqrInv);

            time = (txDist + rxDist) * sosInv + initDel;
            iSamp = time * fs;
            if(iSamp < 0.0f || iSamp >= static_cast<float>(nSamp - 1)) {
                continue;
            }
            offset = txOffset + iRx * nSamp;
            interpWgh = modff(iSamp, &iSamp);
            int intSamp = int(iSamp);

            __sincosf(omega * time, &modSin, &modCos);
            complex<float> modFactor = complex<float>(modCos, modSin);

            samp = iqRaw[offset + intSamp] * (1 - interpWgh) + iqRaw[offset + intSamp + 1] * interpWgh;
            pix += samp * modFactor * rxApod;
            pixWgh += rxApod;
        }
    }
    if(pixWgh == 0.0f) {
        iqLri[z + x*nZPix + iGlobalTx*nZPix*nXPix] = complex<float>(0.0f, 0.0f);
    } else {
        iqLri[z + x*nZPix + iGlobalTx*nZPix*nXPix] = pix / pixWgh * txApod;
    }
}