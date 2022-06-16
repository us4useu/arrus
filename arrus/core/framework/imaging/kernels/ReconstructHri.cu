#ifndef CPP_EXAMPLE_KERNELS_RECONSTRUCT_LRI_PWI_CUH
#define CPP_EXAMPLE_KERNELS_RECONSTRUCT_LRI_PWI_CUH

#include "ReconstructHri.h"

#define _USE_MATH_DEFINES // TODO MSVC specific
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace arrus_example_imaging {

using namespace thrust;

__constant__ float zElemConst[256];
__constant__ float xElemConst[256];
__constant__ float tangElemConst[256];

__global__ void iqRaw2Hri(float2 *iqLri, const float2 *iqRaw, const unsigned nElem, const unsigned nSeq,
                          const unsigned nTx, const unsigned nSamp,
                          const float *zPix, const int nZPix, const float *xPix, const int nXPix, float const sos,
                          float const fs, float const fn, const float *txFoc, const float *txAngZX,
                          const float *txApCentZ, const float *txApCentX, const unsigned *txApFstElem,
                          const unsigned *txApLstElem, const unsigned *rxApOrigElem, const unsigned nRx, const float minRxTang,
                          const float maxRxTang, const float initDel) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int sequenceNr = blockIdx.z * blockDim.z + threadIdx.z;

    if (z >= nZPix || x >= nXPix || sequenceNr >= nSeq) {
        return;
    }

    int iElem, offset;
    float interpWgh;
    float txDist, rxDist, rxTang, txApod, rxApod, time, iSamp;
    float modSin, modCos, pixWgh = 0.0f;
    const float omega = 2 * M_PI * fn;
    const float sosInv = 1 / sos;
    const float nSigma = 3;// number of sigmas in half of the apodization Gaussian curve
    const float twoSigSqrInv = nSigma * nSigma * 0.5f;
    const float rngRxTangInv = 2 / (maxRxTang - minRxTang);// inverted half range
    const float centRxTang = (maxRxTang + minRxTang) * 0.5f;
    complex<float> pix(0.0f, 0.0f), samp(0.0f, 0.0f), modFactor;
    complex<float> a(0.0f, 0.0f), b(0.0f, 0.0f), resultPix(0.0f, 0.0f);
    float2 aRaw, bRaw;

    int sequenceOffset = sequenceNr * nTx * nSamp * nRx;

    for(int iTx = 0; iTx < nTx; ++iTx) {
        if (!isinf(txFoc[iTx])) {
            /* STA */
            float zFoc = txApCentZ[iTx] + txFoc[iTx] * cosf(txAngZX[iTx]);
            float xFoc = txApCentX[iTx] + txFoc[iTx] * sinf(txAngZX[iTx]);

            float pixFocArrang;

            if (txFoc[iTx] <= 0.0f) {
                /* Virtual Point Source BEHIND probe surface */
                // Valid pixels are assumed to be always in front of the focal point (VSP)
                pixFocArrang = 1.0f;
            } else {
                /* Virtual Point Source IN FRONT OF probe surface */
                // Projection of the Foc-Pix vector on the ApCent-Foc vector (dot product) ...
                // to determine if the pixel is behind (-) or in front of (+) the focal point (VSP).
                pixFocArrang =
                    (((zPix[z] - zFoc) * (zFoc - txApCentZ[iTx]) + (xPix[x] - xFoc) * (xFoc - txApCentX[iTx])) >= 0.f)
                    ? 1.f
                    : -1.f;
            }
            txDist = hypotf(zPix[z] - zFoc, xPix[x] - xFoc);
            txDist *= pixFocArrang;// Compensation for the Pix-Foc arrangement
            txDist += txFoc[iTx];  // Compensation for the reference time being the moment when txApCent fires.

            // Projections of Foc-Pix vector on the rotated Foc-ApEdge vectors (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // Foc-ApEdgeFst vector is rotated left, Foc-ApEdgeLst vector is rotated right.
            txApod = (((-(xElemConst[txApFstElem[iTx]] - xFoc) * (zPix[z] - zFoc)
                        + (zElemConst[txApFstElem[iTx]] - zFoc) * (xPix[x] - xFoc))
                           * pixFocArrang
                       >= 0.f)
                      && (((xElemConst[txApLstElem[iTx]] - xFoc) * (zPix[z] - zFoc)
                           - (zElemConst[txApLstElem[iTx]] - zFoc) * (xPix[x] - xFoc))
                              * pixFocArrang
                          >= 0.f))
                ? 1.f
                : 0.f;
        } else {
            /* PWI */
            txDist = (zPix[z] - txApCentZ[iTx]) * cosf(txAngZX[iTx]) + (xPix[x] - txApCentX[iTx]) * sinf(txAngZX[iTx]);

            // Projections of ApEdge-Pix vector on the rotated unit vector of tx direction (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // For ApEdgeFst, the vector is rotated left, for ApEdgeLst the vector is rotated right.
            txApod = (((-(zPix[z] - zElemConst[txApFstElem[iTx]]) * sinf(txAngZX[iTx])
                        + (xPix[x] - xElemConst[txApFstElem[iTx]]) * cosf(txAngZX[iTx]))
                       >= 0.f)
                      && (((zPix[z] - zElemConst[txApLstElem[iTx]]) * sinf(txAngZX[iTx])
                           - (xPix[x] - xElemConst[txApLstElem[iTx]]) * cosf(txAngZX[iTx]))
                          >= 0.f))
                ? 1.f
                : 0.f;
        }
        pixWgh = 0.0f;
        int txOffset = sequenceOffset + iTx*nSamp*nRx;
        if (txApod != 0.0f) {
            for (int iRx = 0; iRx < nRx; iRx++) {
                iElem = iRx + rxApOrigElem[iTx];
                if (iElem < 0 || iElem >= nElem)
                    continue;

                rxDist = hypotf(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
                rxTang = __fdividef(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
                rxTang = __fdividef(rxTang - tangElemConst[iElem], 1.f + rxTang * tangElemConst[iElem]);
                if (rxTang < minRxTang || rxTang > maxRxTang)
                    continue;

                rxApod = (rxTang - centRxTang) * rngRxTangInv;
                rxApod = __expf(-rxApod * rxApod * twoSigSqrInv);

                time = (txDist + rxDist) * sosInv + initDel;
                iSamp = time * fs;
                if (iSamp < 0.0f || iSamp >= static_cast<float>(nSamp - 1)) {
                    continue;
                }
                offset = txOffset + iRx * nSamp;
                interpWgh = modff(iSamp, &iSamp);
                int intSamp = int(iSamp);

                __sincosf(omega * time, &modSin, &modCos);
                modFactor = complex<float>(modCos, modSin);

                aRaw = iqRaw[offset + intSamp];
                bRaw = iqRaw[offset + intSamp + 1];
                a = complex<float>(aRaw.x, aRaw.y);
                b = complex<float>(bRaw.x, bRaw.y);

                samp = a * (1 - interpWgh) + b * interpWgh;

                pix += samp * modFactor * rxApod;
                pixWgh += rxApod;
            }
        }
        if (pixWgh != 0.0f) {
            resultPix += pix / pixWgh * txApod;
        }
    }
    iqLri[z + x * nZPix + sequenceNr * nZPix * nXPix].x = resultPix.real();
    iqLri[z + x * nZPix + sequenceNr * nZPix * nXPix].y = resultPix.imag();
}

ReconstructHriFunctor::ReconstructHriFunctor(const NdArray &zElemPos, const NdArray &xElemPos,
                                             const NdArray &elementTang) {
    CUDA_ASSERT(
        cudaMemcpyToSymbol(zElemConst, zElemPos.getConstPtr<float>(), zElemPos.getNBytes(), 0, cudaMemcpyHostToDevice));
    CUDA_ASSERT(
        cudaMemcpyToSymbol(xElemConst, xElemPos.getConstPtr<float>(), xElemPos.getNBytes(), 0, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpyToSymbol(tangElemConst, elementTang.getConstPtr<float>(), elementTang.getNBytes(), 0,
                                   cudaMemcpyHostToDevice));
}

void ReconstructHriFunctor::operator()(NdArray &output, const NdArray &input, const NdArray &zPix, const NdArray &xPix,
                                       const NdArray &txFocuses, const NdArray &txAngles,
                                       const NdArray &txApertureCenterZ, const NdArray &txApertureCenterX,
                                       const NdArray &txApertureFirstElement, const NdArray &txApertureLastElement,
                                       const NdArray &rxApertureOrigin, const unsigned int nElements, const float sos,
                                       const float fs, const float fn, const float minRxTang, const float maxRxTang,
                                       const float initDelay,
                                       cudaStream_t stream) {
    dim3 block(16, 16, 4);
    unsigned nSequences = input.getShape()[0];
    unsigned nTx = input.getShape()[1];
    unsigned nRx = input.getShape()[2];
    unsigned nSamples = input.getShape()[3];

    unsigned nz = zPix.getShape()[0];
    unsigned nx = xPix.getShape()[0];
    dim3 grid(
        (nz + block.x - 1) / block.x,
        (nx + block.y - 1) / block.y,
        (nSequences + block.z - 1) / block.z
    );
    iqRaw2Hri<<<grid, block, 0, stream>>>(
        output.getPtr<float2>(), input.getConstPtr<float2>(),
        nElements,
        nSequences, nTx, nSamples,
        zPix.getConstPtr<float>(), nz,
        xPix.getConstPtr<float>(), nx,
        sos, fs, fn,
        txFocuses.getConstPtr<float>(),
        txAngles.getConstPtr<float>(),
        txApertureCenterZ.getConstPtr<float>(), txApertureCenterX.getConstPtr<float>(),
        txApertureFirstElement.getConstPtr<unsigned>(), txApertureLastElement.getConstPtr<unsigned>(),
        rxApertureOrigin.getConstPtr<unsigned>(),
        nRx,
        minRxTang, maxRxTang,
        initDelay
    );
    CUDA_ASSERT(cudaGetLastError());
}

}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_KERNELS_RECONSTRUCT_LRI_PWI_CUH
