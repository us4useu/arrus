#include <cupy/complex.cuh>
#define M_PI 3.14159265358979

__constant__ float xElemConst[256];
__constant__ float yElemConst[256];

__forceinline__ __device__ float ownHypotf(float z, float x)
{
    return sqrtf(z*z + x*x);
}

__forceinline__ __device__ float ownHypotf(float z, float x, float y)
{
    return sqrtf(z*z + x*x + y*y);
}

__forceinline__ __device__  complex<float> interpLinear(const complex<float> *input, float sample) {
    float interpWgh = modff(sample, &sample);
    int intSample = int(sample);
    return input[intSample]*(1 - interpWgh) + input[intSample+1]*interpWgh;
}

__forceinline__ __device__  float interpLinearNormalized(const float *input, int inputSize, float sample) {
    // Normalized => sample == 0 means input[0], sample == 1 means input[inputSize-1]
    sample = sample*(inputSize-1);
    float interpWgh = modff(sample, &sample);
    int intSample = int(sample);
    return input[intSample]*(1-interpWgh) + input[intSample+1]*interpWgh;
}

extern "C"
__global__ void iqRaw2Lri3D(complex<float> *iqLri, const complex<float> *input,
                            const int nSeq, const int nTx, const int nElemY, const int nElemX, const int nSamp,
                            const float *zPix, const int nZPix,
                            const float *xPix, const int nXPix,
                            const float *yPix, const int nYPix,
                            const float sos, const float fs, const float fn,
                            const float *txFoc, const float *txAngZX, const float *txAngZY,
                            const float *txApCentX, const float *txApCentY,
                            const int *txApFstElemX, const int *txApLstElemX,
                            const int *txApFstElemY, const int *txApLstElemY,
                            const float minRxTangZX, const float maxRxTangZX,
                            const float minRxTangZY, const float maxRxTangZY,
                            const float initDel,
                            const float *rxApod, const int nRxApod) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;
    if (z >= nZPix || x >= nXPix || y >= nYPix) {
        return;
    }
    unsigned txOffset, offset;
    float txAngZn, txAngAz;
    float txDist, rxDist, rxTang, txApod, rxApodX, rxApodY, time, iSamp;
    float modSin, modCos, pixWgh;
    complex<float> samp, pix;
    complex<float> modFactor;
    const float omega = 2 * M_PI * fn;
    const float sosInv = 1 / sos;
    const float zDistInv = 1 / zPix[z];
    const float rngRxTangZXInv = 1 / (maxRxTangZX - minRxTangZX); // inverted zx-tangent range TODO parameter?
    const float rngRxTangZYInv = 1 / (maxRxTangZY - minRxTangZY); // inverted zy-tangent range TODO parameter?
    iqLri[z + x*nZPix + y*nZPix*nXPix] = 0;

    for (int iTx=0; iTx < nTx; ++iTx) {
        // Plane inclinations -> spherical coordinates
        txAngZn = atanf(ownHypotf(tanf(txAngZX[iTx]), tanf(txAngZY[iTx]))); // Zenith
        txAngAz = atan2f(tanf(txAngZY[iTx]), tanf(txAngZX[iTx])); // Azimuth
        txOffset = iTx*nSamp*nElemX*nElemY; // TODO number of sequences in transmit?

        if (!isinf(txFoc[iTx])) {
            /* STA */
            float zFoc = txFoc[iTx] * cosf(txAngZn);
            float xFoc = txFoc[iTx] * sinf(txAngZn) * cosf(txAngAz) + txApCentX[iTx];
            float yFoc = txFoc[iTx] * sinf(txAngZn) * sinf(txAngAz) + txApCentY[iTx];

            float pixFocArrang;

            if (txFoc[iTx] <= 0.f) {
                /* Virtual Point Source BEHIND probe surface */
                // Valid pixels are assumed to be always in front of the focal point (VSP)
                pixFocArrang = 1.f;
            }
            else {
                /* Virtual Point Source IN FRONT OF probe surface */
                // Projection of the Foc-Pix vector on the ApCent-Foc vector (dot product) ...
                // to determine if the pixel is behind (-) or in front of (+) the focal point (VSP).
                pixFocArrang = (((zPix[z]-zFoc)* zFoc + (xPix[x]-xFoc)*(xFoc-txApCentX[iTx]) + (yPix[x]-yFoc)*(yFoc-txApCentY[iTx])) >= 0.f) ? 1.f : -1.f;
            }
            txDist = ownHypotf(zPix[z]-zFoc, xPix[x]-xFoc, yPix[y]-yFoc);
            txDist *= pixFocArrang; // Compensation for the Pix-Foc arrangement
            // txFoc is an artificial time, that actually does not happen
            txDist += txFoc[iTx]; // Compensation for the reference time being the moment when txApCent fires.

            // Projections of Foc-Pix vector on the rotated Foc-ApEdge vectors (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // Foc-ApEdgeFst vector is rotated left, Foc-ApEdgeLst vector is rotated right.
            txApod = (
                    ((-(zPix[z] - zFoc)*(xElemConst[txApFstElemX[iTx]] - xFoc) - (xPix[x] - xFoc)*zFoc) * pixFocArrang >= 0.f) &&
                    ( ((zPix[z] - zFoc)*(xElemConst[txApLstElemX[iTx]] - xFoc) + (xPix[x] - xFoc)*zFoc) * pixFocArrang >= 0.f) &&
                    ((-(zPix[z] - zFoc)*(yElemConst[txApFstElemY[iTx]] - yFoc) - (yPix[y] - yFoc)*zFoc) * pixFocArrang >= 0.f) &&
                    (( (zPix[z] - zFoc)*(yElemConst[txApLstElemY[iTx]] - yFoc) + (yPix[y] - yFoc)*zFoc) * pixFocArrang >= 0.f)
                    ) ? 1.f : 0.f;
        }
        else {
            /* PWI */
            txDist = zPix[z] * cosf(txAngZn) + ((xPix[x] - txApCentX[iTx]) * cosf(txAngAz) + (yPix[y] - txApCentY[iTx]) * sinf(txAngAz)) * sinf(txAngZn);

            // Projections of ApEdge-Pix vector on the rotated unit vector of tx direction (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // For ApEdgeFst, the vector is rotated left, for ApEdgeLst the vector is rotated right.
            txApod = (((-sinf(txAngZX[iTx])*zPix[z] + cosf(txAngZX[iTx]) * (xPix[x]-xElemConst[txApFstElemX[iTx]])) >= 0.f) &&
                    (   (sinf(txAngZX[iTx])*zPix[z] - cosf(txAngZX[iTx]) * (xPix[x]-xElemConst[txApLstElemX[iTx]])) >= 0.f) &&
                    (  (-sinf(txAngZY[iTx])*zPix[z] + cosf(txAngZY[iTx]) * (yPix[x]-yElemConst[txApFstElemY[iTx]])) >= 0.f) &&
                    (   (sinf(txAngZY[iTx])*zPix[z] - cosf(txAngZY[iTx]) * (yPix[x]-yElemConst[txApLstElemY[iTx]])) >= 0.f )) ? 1.f : 0.f;
        }

        pix.real(0.0f);
        pix.imag(0.0f);
        pixWgh = 0.0f;

        if (txApod != 0.f) {
            for (int iElemY = 0; iElemY < nElemY; ++iElemY) {
                rxTang = (yPix[y]-yElemConst[iElemY])*zDistInv;
                if (rxTang < minRxTangZY || rxTang > maxRxTangZY) continue;
                rxApodY = interpLinearNormalized(rxApod, nRxApod, (rxTang-minRxTangZY)*rngRxTangZYInv);
                for (int iElemX = 0; iElemX < nElemX; ++iElemX) {
                    offset = iTx*nElemY*nElemX*nSamp + iElemY*nElemX*nSamp + iElemX*nSamp;
                    rxTang = (xPix[x] - xElemConst[iElemX])*zDistInv;
                    if (rxTang < minRxTangZX || rxTang > maxRxTangZX) continue;
                    rxApodX = interpLinearNormalized(rxApod, nRxApod, (rxTang-minRxTangZX)*rngRxTangZXInv);
                    rxDist = ownHypotf(zPix[z], xPix[x] - xElemConst[iElemX], yPix[y] - yElemConst[iElemY]);
                    time = (txDist+rxDist)*sosInv + initDel;
                    iSamp = time*fs;
                    if (iSamp < 0 || iSamp > static_cast<float>(nSamp-1)) continue;
                    __sincosf(omega*time, &modSin, &modCos);
                    modFactor = complex<float>(modCos, modSin);
                    samp = interpLinear(input+offset, iSamp);
                    pix += samp*modFactor*rxApodX*rxApodY;
                    pixWgh += rxApodX*rxApodY;
                }
            }
        }
        if (pixWgh != 0.0f) {
            iqLri[z + x*nZPix + y*nZPix*nXPix] += pix * txApod / pixWgh;
        }
    }
}

