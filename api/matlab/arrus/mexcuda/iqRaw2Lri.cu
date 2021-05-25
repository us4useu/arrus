#define M_PI 3.14159265358979
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <iostream>

__constant__ float xElemConst[1024];

texture <float2, cudaTextureType1DLayered, cudaReadModeElementType> iqRawTex;


__forceinline__ __device__ float ownHypotf(float x, float y)
{
    return sqrtf(x*x + y*y);
}


__global__ void iqRaw2Lri(  float2 * iqLri, float const * zPix, float const * xPix, 
                            float const * txFoc, float const * txAng, 
                            float const * txApCentZ, float const * txApCentX, 
                            int const * txApFstElem, int const * txApLstElem, 
                            float const minRxTang, float const maxRxTang, 
                            float const fs, float const fn, 
                            float const sos, float const initDel, 
                            int const nZPix, int const nXPix, 
                            int const nSamp, int const nElem, 
                            int const nTx)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    float txDist, rxDist, rxTang, txApod, rxApod, time, iSamp;
    float modSin, modCos, sampRe, sampIm, pixRe, pixIm, pixWgh;
    float const omega = 2 * M_PI * fn;
    float const sosInv = 1 / sos;
//     float const zDistInv = 1 / zPix[z];
    float const nSigma = 3; // number of sigmas in half of the apodization Gaussian curve
    float const twoSigSqrInv = nSigma * nSigma * 0.5f;
    float const rngRxTangInv = 2 / (maxRxTang - minRxTang); // inverted half range
    float const centRxTang = (maxRxTang + minRxTang) * 0.5f;
    
    for (int iTx=0; iTx<nTx; iTx++) {
        
        if (!isinf(txFoc[iTx])) {
            /* STA */
            float zFoc	= txApCentZ[iTx] + txFoc[iTx] * cosf(txAng[iTx]);
            float xFoc	= txApCentX[iTx] + txFoc[iTx] * sinf(txAng[iTx]);
            
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
                pixFocArrang = (((zPix[z]-zFoc)*(zFoc-txApCentZ[iTx]) + 
                                 (xPix[x]-xFoc)*(xFoc-txApCentX[iTx])) >= 0.f) ? 1.f : -1.f;
            }
            txDist	= ownHypotf(zPix[z] - zFoc, xPix[x] - xFoc);
            txDist *= pixFocArrang; // Compensation for the Pix-Foc arrangement
            txDist += txFoc[iTx]; // Compensation for the reference time being the moment when txApCent fires.
            
            // Projections of Foc-Pix vector on the rotated Foc-ApEdge vectors (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // Foc-ApEdgeFst vector is rotated left, Foc-ApEdgeLst vector is rotated right.
            txApod = ( ( (-(xElemConst[txApFstElem[iTx]] - xFoc)*(zPix[z] - zFoc) + 
                           (zElemConst[txApFstElem[iTx]] - zFoc)*(xPix[x] - xFoc))*pixFocArrang >= 0.f ) && 
                       ( ( (xElemConst[txApLstElem[iTx]] - xFoc)*(zPix[z] - zFoc) - 
                           (zElemConst[txApLstElem[iTx]] - zFoc)*(xPix[x] - xFoc))*pixFocArrang >= 0.f ) ) ? 1.f : 0.f;
        }
        else {
            /* PWI */
            txDist = (zPix[z] - txApCentZ[iTx]) * cosf(txAng[iTx]) + 
                     (xPix[x] - txApCentX[iTx]) * sinf(txAng[iTx]);
            
            // Projections of ApEdge-Pix vector on the rotated unit vector of tx direction (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // For ApEdgeFst, the vector is rotated left, for ApEdgeLst the vector is rotated right.
            txApod = ( ( (-(zPix[z]-zElemConst[txApFstElem[iTx]]) * sinf(txAng[iTx]) + 
                           (xPix[x]-xElemConst[txApFstElem[iTx]]) * cosf(txAng[iTx])) >= 0.f ) && 
                       ( ( (zPix[z]-zElemConst[txApLstElem[iTx]]) * sinf(txAng[iTx]) - 
                           (xPix[x]-xElemConst[txApLstElem[iTx]]) * cosf(txAng[iTx])) >= 0.f ) ) ? 1.f : 0.f;
        }
        
        pixRe = 0.f;
        pixIm = 0.f;
        pixWgh = 0.f;
        
        if (txApod != 0.f) {
            for (int iElem=0; iElem<nElem; iElem++) {
                rxDist = ownHypotf(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
//                 rxTang = (xPix[x] - xElemConst[iElem]) * zDistInv;
                rxTang = __fdividef(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
                rxTang = __fdividef(rxTang-tangElemConst[iElem], 1.f+rxTang*tangElemConst[iElem]);
                if (rxTang < minRxTang || rxTang > maxRxTang) continue;
                rxApod = (rxTang-centRxTang)*rngRxTangInv;
                rxApod = __expf(-rxApod*rxApod*twoSigSqrInv);
                
                time = (txDist + rxDist) * sosInv + initDel;
                iSamp = time * fs;
                if (iSamp<0.f || iSamp>static_cast<float>(nSamp-1)) continue;
                
                float2 iqSamp = tex1DLayered(iqRawTex, iSamp + 0.5f, iElem + iTx*nElem);
                sampRe = iqSamp.x;
                sampIm = iqSamp.y;
                
                __sincosf(omega * time, &modSin, &modCos);
                
                pixRe += (sampRe * modCos - sampIm * modSin) * rxApod; // 60-80us
                pixIm += (sampRe * modSin + sampIm * modCos) * rxApod;
                pixWgh += rxApod;
            }
        }
        
        iqLri[z + x*nZPix + iTx*nZPix*nXPix].x = pixRe / pixWgh * txApod;
        iqLri[z + x*nZPix + iTx*nZPix*nXPix].y = pixIm / pixWgh * txApod;
    }
}

__host__ void checkData(mxGPUArray const * const data, char const * const name, bool const isComplex, int const nDims, char const * const invalidInputMsgId)
{
    std::string invalidInputMsgTxt(name);
    
    if (mxGPUGetClassID(data) != mxSINGLE_CLASS) 
        invalidInputMsgTxt += std::string(" must be single.");
    
    else if (!isComplex && mxGPUGetComplexity(data)) 
        invalidInputMsgTxt += std::string(" must be real.");
    
    else if (isComplex && !mxGPUGetComplexity(data)) 
        invalidInputMsgTxt += std::string(" must be complex.");
    
    else if (nDims==1 && !( mxGPUGetNumberOfDimensions(data) == 1 || 
                           (mxGPUGetNumberOfDimensions(data) == 2 && mxGPUGetDimensions(data)[0] == 1))) 
        invalidInputMsgTxt += std::string(" must be at most 1D vector.");
    
    else if (nDims==2 && !(mxGPUGetNumberOfDimensions(data) <= 2)) 
        invalidInputMsgTxt += std::string(" must be at most 2D array.");
    
    else if (nDims==3 && !(mxGPUGetNumberOfDimensions(data) <= 3)) 
        invalidInputMsgTxt += std::string(" must be at most 3D array.");
    
    else
        return;
    
    std::cout << " " << std::endl; // This line prevents crash, no idea why?
    mexErrMsgIdAndTxt( invalidInputMsgId, invalidInputMsgTxt.c_str());
}

void mexFunction(int nlhs, mxArray * plhs[],
                 int nrhs, mxArray const * prhs[])
{
    /* Initialize the GPU API. */
    mxInitGPU();
    
    /* Declare the variables */
    mxGPUArray * iqLri;
    mxGPUArray const * iqRaw;
    mxGPUArray const * xElem;
    mxGPUArray const * zPix;
    mxGPUArray const * xPix;
    mxGPUArray const * foc;
    mxGPUArray const * ang;
    mxGPUArray const * cent;
    
    float2 * dev_iqLri;
    float2 const * dev_iqRaw;
    float const * dev_xElem;
    float const * dev_zPix;
    float const * dev_xPix;
    float const * dev_foc;
    float const * dev_ang;
    float const * dev_cent;
    
    float minRxTang;
    float maxRxTang;
    float fs;
    float fn;
    float sos;
    float initDel;
    
    int nSamp;
    int nElem;
    int nZPix;
    int nXPix;
    int nTx;
    
    dim3 const threadsPerBlock = {16, 16, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
    char const * const invalidInputMsgId = "iqRaw2Lri:InvalidInput";
    char const * const invalidOutputMsgId = "iqRaw2Lri:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=13) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "13 inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }
    
//     for (int i=7; i<13; i++) {
//         if (!mxIsSingle(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
//             mexErrMsgIdAndTxt( invalidInputMsgId, "Last 6 inputs must be single, real scalars");
//         }
//     }
    
    
    /* Extract inputs from prhs */
    iqRaw = mxGPUCreateFromMxArray(prhs[0]);
    xElem = mxGPUCreateFromMxArray(prhs[1]);
    zPix  = mxGPUCreateFromMxArray(prhs[2]);
    xPix  = mxGPUCreateFromMxArray(prhs[3]);
    foc   = mxGPUCreateFromMxArray(prhs[4]);
    ang   = mxGPUCreateFromMxArray(prhs[5]);
    cent  = mxGPUCreateFromMxArray(prhs[6]);
    
    minRxTang = mxGetScalar(prhs[7]);
    maxRxTang = mxGetScalar(prhs[8]);
    fs    = mxGetScalar(prhs[9]);
    fn    = mxGetScalar(prhs[10]);
    sos   = mxGetScalar(prhs[11]);
    initDel	= mxGetScalar(prhs[12]);
    
    /* Validate inputs */
    checkData(iqRaw,     "iqRaw",     true,  3, invalidInputMsgId);
    checkData(xElem,     "xElem",     false, 1, invalidInputMsgId);
    checkData(zPix,      "zPix",      false, 1, invalidInputMsgId);
    checkData(xPix,      "xPix",      false, 1, invalidInputMsgId);
    checkData(foc,       "foc",       false, 1, invalidInputMsgId);
    checkData(ang,       "ang",       false, 1, invalidInputMsgId);
    checkData(cent,      "cent",      false, 1, invalidInputMsgId);
    
    if (mxGPUGetDimensions(iqRaw)[1] != mxGPUGetNumberOfElements(xElem)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "size(iqRaw,2) must be equal to length(xElem).");
    }
    
    /* Get some additional information */
    nSamp = mxGPUGetDimensions(iqRaw)[0];
    nElem = mxGPUGetNumberOfElements(xElem);
    nZPix = mxGPUGetNumberOfElements(zPix);
    nXPix = mxGPUGetNumberOfElements(xPix);
    if (mxGPUGetNumberOfDimensions(iqRaw)<3) {
        nTx = 1;
    }
    else {
        nTx   = mxGPUGetDimensions(iqRaw)[2];
    }
    
    sharedPerBlock = 0;
    blocksPerGrid = {(nZPix+threadsPerBlock.x-1)/threadsPerBlock.x, 
                     (nXPix+threadsPerBlock.y-1)/threadsPerBlock.y, 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 3;
    mwSize dimOut[3] = {nZPix, nXPix, nTx};
    
    iqLri = mxGPUCreateGPUArray(nDimOut,
                                dimOut,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_iqLri = static_cast<float2 *>(mxGPUGetData(iqLri));
    dev_iqRaw = static_cast<float2 const *>(mxGPUGetDataReadOnly(iqRaw));
    dev_xElem = static_cast<float const *>(mxGPUGetDataReadOnly(xElem));
    dev_zPix  = static_cast<float const *>(mxGPUGetDataReadOnly(zPix));
    dev_xPix  = static_cast<float const *>(mxGPUGetDataReadOnly(xPix));
    dev_foc   = static_cast<float const *>(mxGPUGetDataReadOnly(foc));
    dev_ang   = static_cast<float const *>(mxGPUGetDataReadOnly(ang));
    dev_cent  = static_cast<float const *>(mxGPUGetDataReadOnly(cent));
    
    /* set constant memory */
    if(nElem > 1024) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "xElem is too long, kernel supports xElem of up to 1024 elements");
    }
    cudaMemcpyToSymbol(xElemConst, dev_xElem, nElem*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    /* configure texture reference */
    iqRawTex.normalized  = false;
    iqRawTex.addressMode[0] = cudaAddressModeBorder;
    iqRawTex.filterMode  = cudaFilterModeLinear;
    
    int nTxPerPart = (nElem*nTx <= 2048) ? nTx : 2048/nElem;
    int nPart = (nTx+nTxPerPart-1)/nTxPerPart;
    
    cudaArray* cuArray;
    cudaExtent cuArraySize =  make_cudaExtent(nSamp, 0, nElem*nTxPerPart);
    cudaMalloc3DArray(&cuArray, &iqRawTex.channelDesc, cuArraySize, cudaArrayLayered);
    cudaBindTextureToArray(iqRawTex, cuArray);
    
    /* Kernel in loop - due to limited number of texture layers */
    cudaMemcpy3DParms cuArrayCopy = {0};
    cuArrayCopy.dstArray = cuArray;
    cuArrayCopy.kind = cudaMemcpyDeviceToDevice;
    for (int iPart=0; iPart<nPart; iPart++) {
        
        int nTxInThisPart = (iPart<(nPart-1)) ? nTxPerPart : (nTx-iPart*nTxPerPart);
        
        /* Prepare texture memory */
        cuArrayCopy.srcPtr = make_cudaPitchedPtr(const_cast<float2 *>(dev_iqRaw)+iPart*nSamp*nElem*nTxPerPart, nSamp * sizeof(float2), nSamp, 1);
        cuArrayCopy.extent = make_cudaExtent(nSamp, 1, nElem*nTxInThisPart);
        cudaMemcpy3D(&cuArrayCopy);
        
        /* Execute CUDA kernel */
        iqRaw2Lri<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqLri + iPart*nZPix*nXPix*nTxPerPart, 
                                                                      dev_zPix, dev_xPix, 
                                                                      dev_foc       + iPart*nTxPerPart, 
                                                                      dev_ang       + iPart*nTxPerPart, 
                                                                      dev_cent      + iPart*nTxPerPart, 
                                                                      minRxTang, maxRxTang, fs, fn, sos, initDel, 
                                                                      nZPix, nXPix, nSamp, nElem, nTxInThisPart);
        
    }
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(iqLri);
    
    /* Clean-up */
    cudaUnbindTexture(iqRawTex);
    cudaFreeArray(cuArray);
    
    mxGPUDestroyGPUArray(iqLri);
    mxGPUDestroyGPUArray(iqRaw);
    mxGPUDestroyGPUArray(xElem);
    mxGPUDestroyGPUArray(zPix);
    mxGPUDestroyGPUArray(xPix);
    mxGPUDestroyGPUArray(foc);
    mxGPUDestroyGPUArray(ang);
    mxGPUDestroyGPUArray(cent);
    
    //cudaDeviceReset();
}
