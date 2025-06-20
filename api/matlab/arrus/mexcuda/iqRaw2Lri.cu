#define M_PI 3.14159265358979
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <iostream>

__constant__ float zElemConst[256];
__constant__ float xElemConst[256];
__constant__ float tangElemConst[256];

texture <float2, cudaTextureType1DLayered, cudaReadModeElementType> iqRawTex;
texture <float, cudaTextureType1D, cudaReadModeElementType> rxApodTex;

__forceinline__ __device__ float ownHypotf(float x, float y)
{
    return sqrtf(x*x + y*y);
}


__global__ void iqRaw2Lri(  float2 * iqLri, 
                            float const * zPix, 
                            float const * xPix, 
                            float const * txFoc, 
                            float const * txAngZX, 
                            float const * txApCentZ, 
                            float const * txApCentX, 
                            float const * fn, 
                            float const * initDel, 
                            int const * txApFstElem, 
                            int const * txApLstElem, 
                            int const * rxApOrigElem, 
                            int const * nSampOmit, 
                            float const * minRxTang, 
                            float const * maxRxTang, 
                            float const fs, 
                            float const sos, 
                            int const nZPix, 
                            int const nXPix, 
                            int const nSamp, 
                            int const nElem, 
                            int const nRx, 
                            int const nTx)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    int iElem;
    float txDist, rxDist, rxTang, txApod, rxApod, time, iSamp;
    float modSin, modCos, sampRe, sampIm, pixRe, pixIm, pixWgh;
    float const sosInv = 1 / sos;
//     float const zDistInv = 1 / zPix[z];
    
    for (int iTx=0; iTx<nTx; iTx++) {
        
        iElem = (rxApOrigElem[iTx] + nRx <= nElem) ? rxApOrigElem[iTx] + nRx - 1 : nElem - 1;
        rxTang = __fdividef(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
        rxTang = __fdividef(rxTang - tangElemConst[iElem], 1.f + rxTang*tangElemConst[iElem]);
        float minRxTangPix = fmax(minRxTang[iTx], rxTang);
        
        iElem = (rxApOrigElem[iTx] >= 0) ? rxApOrigElem[iTx] : 0;
        rxTang = __fdividef(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
        rxTang = __fdividef(rxTang - tangElemConst[iElem], 1.f + rxTang*tangElemConst[iElem]);
        float maxRxTangPix = fmin(maxRxTang[iTx], rxTang);
        
        float const rngRxTangInv = 1 / (maxRxTangPix - minRxTangPix); // inverted tangent range
        float omega = 2 * M_PI * fn[iTx];
        
        if (!isinf(txFoc[iTx])) {
            /* STA */
            float zFoc	= txApCentZ[iTx] + txFoc[iTx] * cosf(txAngZX[iTx]);
            float xFoc	= txApCentX[iTx] + txFoc[iTx] * sinf(txAngZX[iTx]);
            
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
            txDist = (zPix[z] - txApCentZ[iTx]) * cosf(txAngZX[iTx]) + 
                     (xPix[x] - txApCentX[iTx]) * sinf(txAngZX[iTx]);
            
            // Projections of ApEdge-Pix vector on the rotated unit vector of tx direction (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // For ApEdgeFst, the vector is rotated left, for ApEdgeLst the vector is rotated right.
            txApod = ( ( (-(zPix[z]-zElemConst[txApFstElem[iTx]]) * sinf(txAngZX[iTx]) + 
                           (xPix[x]-xElemConst[txApFstElem[iTx]]) * cosf(txAngZX[iTx])) >= 0.f ) && 
                       ( ( (zPix[z]-zElemConst[txApLstElem[iTx]]) * sinf(txAngZX[iTx]) - 
                           (xPix[x]-xElemConst[txApLstElem[iTx]]) * cosf(txAngZX[iTx])) >= 0.f ) ) ? 1.f : 0.f;
        }
        
        pixRe = 0.f;
        pixIm = 0.f;
        pixWgh = 0.f;
        
        if (txApod != 0.f) {
            for (int iRx=0; iRx<nRx; iRx++) {
                iElem = iRx + rxApOrigElem[iTx];
                if (iElem<0 || iElem>=nElem) continue;
                
                rxDist = ownHypotf(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
//                 rxTang = (xPix[x] - xElemConst[iElem]) * zDistInv;
                rxTang = __fdividef(xPix[x] - xElemConst[iElem], zPix[z] - zElemConst[iElem]);
                rxTang = __fdividef(rxTang-tangElemConst[iElem], 1.f+rxTang*tangElemConst[iElem]);
                if (rxTang < minRxTangPix || rxTang > maxRxTangPix) continue;
                rxApod = (rxTang-minRxTangPix)*rngRxTangInv; // <0,1>, needs normalized texture fetching, errors at aperture sided
                rxApod = tex1D(rxApodTex, rxApod);
                
                time = (txDist + rxDist) * sosInv + initDel[iTx];
                iSamp = time * fs;
                if (iSamp<static_cast<float>(nSampOmit[iTx]) || iSamp>static_cast<float>(nSamp-1)) continue;
                
                float2 iqSamp = tex1DLayered(iqRawTex, iSamp + 0.5f, iRx + iTx*nRx);
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

__host__ void checkData(mxGPUArray const * const data, 
                        char const * const name, 
                        bool const mustBeInt, 
                        bool const mustBeComplex, 
                        int const mustBeNDim, 
                        char const * const invalidInputMsgId)
{
    std::string invalidInputMsgTxt(name);
    
    if (mustBeInt && mxGPUGetClassID(data) != mxINT32_CLASS) 
        invalidInputMsgTxt += std::string(" must be int32.");
    
    else if (!mustBeInt && mxGPUGetClassID(data) != mxSINGLE_CLASS) 
        invalidInputMsgTxt += std::string(" must be single.");
    
    else if (!mustBeComplex && mxGPUGetComplexity(data)) 
        invalidInputMsgTxt += std::string(" must be real.");
    
    else if (mustBeComplex && !mxGPUGetComplexity(data)) 
        invalidInputMsgTxt += std::string(" must be complex.");
    
    else if (mustBeNDim==1 && !( mxGPUGetNumberOfDimensions(data) == 1 || 
                                (mxGPUGetNumberOfDimensions(data) == 2 && 
                                 mxGPUGetDimensions(data)[0] == 1))) 
        invalidInputMsgTxt += std::string(" must be at most 1D vector.");
    
    else if (mustBeNDim==2 && !(mxGPUGetNumberOfDimensions(data) <= 2)) 
        invalidInputMsgTxt += std::string(" must be at most 2D array.");
    
    else if (mustBeNDim==3 && !(mxGPUGetNumberOfDimensions(data) <= 3)) 
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
    mxGPUArray const * zElem;
    mxGPUArray const * xElem;
    mxGPUArray const * tangElem;
    mxGPUArray const * zPix;
    mxGPUArray const * xPix;
    mxGPUArray const * rxApod;
    mxGPUArray const * foc;
    mxGPUArray const * ang;
    mxGPUArray const * centZ;
    mxGPUArray const * centX;
    mxGPUArray const * fn;
    mxGPUArray const * initDel;
    mxGPUArray const * elemFst;
    mxGPUArray const * elemLst;
    mxGPUArray const * rxElemOrig;
    mxGPUArray const * nSampOmit;
    mxGPUArray const * minRxTang;
    mxGPUArray const * maxRxTang;
    
    float2 * dev_iqLri;
    float2 const * dev_iqRaw;
    float const * dev_zElem;
    float const * dev_xElem;
    float const * dev_tangElem;
    float const * dev_zPix;
    float const * dev_xPix;
    float const * dev_rxApod;
    float const * dev_foc;
    float const * dev_ang;
    float const * dev_centZ;
    float const * dev_centX;
    float const * dev_fn;
    float const * dev_initDel;
    int const * dev_elemFst;
    int const * dev_elemLst;
    int const * dev_rxElemOrig;
    int const * dev_nSampOmit;
    float const * dev_minRxTang;
    float const * dev_maxRxTang;
    
    float fs;
    float sos;
    
    int nSamp;
    int nElem;
    int nZPix;
    int nXPix;
    int nRx;
    int nTx;
    int nRxApodSamp;
    
    dim3 const threadsPerBlock = {16, 16, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
    char const * const invalidInputMsgId = "iqRaw2Lri:InvalidInput";
    char const * const invalidOutputMsgId = "iqRaw2Lri:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=21) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "21 inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }
    
//     for (int i=19; i<21; i++) {
//         if (!mxIsSingle(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
//             mexErrMsgIdAndTxt( invalidInputMsgId, "Last 2 inputs must be single, real scalars");
//         }
//     }
    
    
    /* Extract inputs from prhs */
    iqRaw     = mxGPUCreateFromMxArray(prhs[0]);
    zElem     = mxGPUCreateFromMxArray(prhs[1]);
    xElem     = mxGPUCreateFromMxArray(prhs[2]);
    tangElem  = mxGPUCreateFromMxArray(prhs[3]);
    zPix      = mxGPUCreateFromMxArray(prhs[4]);
    xPix      = mxGPUCreateFromMxArray(prhs[5]);
    rxApod    = mxGPUCreateFromMxArray(prhs[6]);
    foc       = mxGPUCreateFromMxArray(prhs[7]);
    ang       = mxGPUCreateFromMxArray(prhs[8]);
    centZ     = mxGPUCreateFromMxArray(prhs[9]);
    centX     = mxGPUCreateFromMxArray(prhs[10]);
    fn        = mxGPUCreateFromMxArray(prhs[11]);
    initDel   = mxGPUCreateFromMxArray(prhs[12]);
    elemFst   = mxGPUCreateFromMxArray(prhs[13]);
    elemLst   = mxGPUCreateFromMxArray(prhs[14]);
    rxElemOrig= mxGPUCreateFromMxArray(prhs[15]);
    nSampOmit = mxGPUCreateFromMxArray(prhs[16]);
    minRxTang = mxGPUCreateFromMxArray(prhs[17]);
    maxRxTang = mxGPUCreateFromMxArray(prhs[18]);

    fs        = mxGetScalar(prhs[19]);
    sos       = mxGetScalar(prhs[20]);
    
    /* Validate inputs */
    checkData(iqRaw,     "iqRaw",     false, true,  3, invalidInputMsgId);
    checkData(zElem,     "zElem",     false, false, 1, invalidInputMsgId);
    checkData(xElem,     "xElem",     false, false, 1, invalidInputMsgId);
    checkData(tangElem,  "tangElem",  false, false, 1, invalidInputMsgId);
    checkData(zPix,      "zPix",      false, false, 1, invalidInputMsgId);
    checkData(xPix,      "xPix",      false, false, 1, invalidInputMsgId);
    checkData(rxApod,    "rxApod",    false, false, 1, invalidInputMsgId);
    checkData(foc,       "foc",       false, false, 1, invalidInputMsgId);
    checkData(ang,       "ang",       false, false, 1, invalidInputMsgId);
    checkData(centZ,     "centZ",     false, false, 1, invalidInputMsgId);
    checkData(centX,     "centX",     false, false, 1, invalidInputMsgId);
    checkData(fn,        "fn",        false, false, 1, invalidInputMsgId);
    checkData(initDel,   "initDel",   false, false, 1, invalidInputMsgId);
    checkData(elemFst,   "elemFst",   true,  false, 1, invalidInputMsgId);
    checkData(elemLst,   "elemLst",   true,  false, 1, invalidInputMsgId);
    checkData(rxElemOrig,"rxElemOrig",true,  false, 1, invalidInputMsgId);
    checkData(nSampOmit, "nSampOmit", true,  false, 1, invalidInputMsgId);
    checkData(minRxTang, "minRxTang", false, false, 1, invalidInputMsgId);
    checkData(maxRxTang, "maxRxTang", false, false, 1, invalidInputMsgId);
    
    /* Get some additional information */
    nSamp = mxGPUGetDimensions(iqRaw)[0];
    nRx   = mxGPUGetDimensions(iqRaw)[1];
    nElem = mxGPUGetNumberOfElements(xElem);
    nZPix = mxGPUGetNumberOfElements(zPix);
    nXPix = mxGPUGetNumberOfElements(xPix);
    nRxApodSamp = mxGPUGetNumberOfElements(rxApod);
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
    dev_iqLri    = static_cast<float2 *>(mxGPUGetData(iqLri));
    dev_iqRaw    = static_cast<float2 const *>(mxGPUGetDataReadOnly(iqRaw));
    dev_zElem    = static_cast<float const *>(mxGPUGetDataReadOnly(zElem));
    dev_xElem    = static_cast<float const *>(mxGPUGetDataReadOnly(xElem));
    dev_tangElem = static_cast<float const *>(mxGPUGetDataReadOnly(tangElem));
    dev_zPix     = static_cast<float const *>(mxGPUGetDataReadOnly(zPix));
    dev_xPix     = static_cast<float const *>(mxGPUGetDataReadOnly(xPix));
    dev_rxApod   = static_cast<float const *>(mxGPUGetDataReadOnly(rxApod));
    dev_foc      = static_cast<float const *>(mxGPUGetDataReadOnly(foc));
    dev_ang      = static_cast<float const *>(mxGPUGetDataReadOnly(ang));
    dev_centZ    = static_cast<float const *>(mxGPUGetDataReadOnly(centZ));
    dev_centX    = static_cast<float const *>(mxGPUGetDataReadOnly(centX));
    dev_fn       = static_cast<float const *>(mxGPUGetDataReadOnly(fn));
    dev_initDel  = static_cast<float const *>(mxGPUGetDataReadOnly(initDel));
    dev_elemFst  = static_cast<int const *>(mxGPUGetDataReadOnly(elemFst));
    dev_elemLst  = static_cast<int const *>(mxGPUGetDataReadOnly(elemLst));
    dev_rxElemOrig = static_cast<int const *>(mxGPUGetDataReadOnly(rxElemOrig));
    dev_nSampOmit  = static_cast<int const *>(mxGPUGetDataReadOnly(nSampOmit));
    dev_minRxTang  = static_cast<float const *>(mxGPUGetDataReadOnly(minRxTang));
    dev_maxRxTang  = static_cast<float const *>(mxGPUGetDataReadOnly(maxRxTang));
    
    /* set constant memory */
    if(nElem > 256) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "xElem is too long, kernel supports xElem of up to 256 elements");
    }
    cudaMemcpyToSymbol(   zElemConst, dev_zElem,    nElem*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(   xElemConst, dev_xElem,    nElem*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(tangElemConst, dev_tangElem, nElem*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    /* configure texture reference (apodization) */
    rxApodTex.normalized = true;
    rxApodTex.addressMode[0] = cudaAddressModeBorder;
    rxApodTex.filterMode = cudaFilterModeLinear;
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArrayApod;
    cudaMallocArray(&cuArrayApod, &channelDesc, nRxApodSamp, 0);
    cudaMemcpyToArray(cuArrayApod, 0, 0, dev_rxApod, nRxApodSamp*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaBindTextureToArray(rxApodTex, cuArrayApod, channelDesc);
    
    /* configure texture reference */
    iqRawTex.normalized  = false;
    iqRawTex.addressMode[0] = cudaAddressModeBorder;
    iqRawTex.filterMode  = cudaFilterModeLinear;
    
    int nTxPerPart = (nRx*nTx <= 2048) ? nTx : 2048/nRx;
    int nPart = (nTx+nTxPerPart-1)/nTxPerPart;
    
    cudaArray* cuArray;
    cudaExtent cuArraySize =  make_cudaExtent(nSamp, 0, nRx*nTxPerPart);
    cudaMalloc3DArray(&cuArray, &iqRawTex.channelDesc, cuArraySize, cudaArrayLayered);
    cudaBindTextureToArray(iqRawTex, cuArray);
    
    /* Kernel in loop - due to limited number of texture layers */
    cudaMemcpy3DParms cuArrayCopy = {0};
    cuArrayCopy.dstArray = cuArray;
    cuArrayCopy.kind = cudaMemcpyDeviceToDevice;
    for (int iPart=0; iPart<nPart; iPart++) {
        
        int nTxInThisPart = (iPart<(nPart-1)) ? nTxPerPart : (nTx-iPart*nTxPerPart);
        
        /* Prepare texture memory */
        cuArrayCopy.srcPtr = make_cudaPitchedPtr(const_cast<float2 *>(dev_iqRaw)+iPart*nSamp*nRx*nTxPerPart, nSamp * sizeof(float2), nSamp, 1);
        cuArrayCopy.extent = make_cudaExtent(nSamp, 1, nRx*nTxInThisPart);
        cudaMemcpy3D(&cuArrayCopy);
        
        /* Execute CUDA kernel */
        iqRaw2Lri<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqLri + iPart*nZPix*nXPix*nTxPerPart, 
                                                                      dev_zPix, 
                                                                      dev_xPix, 
                                                                      dev_foc       + iPart*nTxPerPart, 
                                                                      dev_ang       + iPart*nTxPerPart, 
                                                                      dev_centZ     + iPart*nTxPerPart, 
                                                                      dev_centX     + iPart*nTxPerPart, 
                                                                      dev_fn        + iPart*nTxPerPart, 
                                                                      dev_initDel   + iPart*nTxPerPart, 
                                                                      dev_elemFst   + iPart*nTxPerPart, 
                                                                      dev_elemLst   + iPart*nTxPerPart, 
                                                                      dev_rxElemOrig + iPart*nTxPerPart, 
                                                                      dev_nSampOmit + iPart*nTxPerPart, 
                                                                      dev_minRxTang + iPart*nTxPerPart, 
                                                                      dev_maxRxTang + iPart*nTxPerPart, 
                                                                      fs, sos, 
                                                                      nZPix, nXPix, nSamp, nElem, nRx, nTxInThisPart);
    }
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(iqLri);
    
    /* Clean-up */
    cudaUnbindTexture(iqRawTex);
    cudaFreeArray(cuArray);
    
    cudaUnbindTexture(rxApodTex);
    cudaFreeArray(cuArrayApod);
    
    mxGPUDestroyGPUArray(iqLri);
    mxGPUDestroyGPUArray(iqRaw);
    mxGPUDestroyGPUArray(zElem);
    mxGPUDestroyGPUArray(xElem);
    mxGPUDestroyGPUArray(tangElem);
    mxGPUDestroyGPUArray(zPix);
    mxGPUDestroyGPUArray(xPix);
    mxGPUDestroyGPUArray(rxApod);
    mxGPUDestroyGPUArray(foc);
    mxGPUDestroyGPUArray(ang);
    mxGPUDestroyGPUArray(centZ);
    mxGPUDestroyGPUArray(centX);
    mxGPUDestroyGPUArray(fn);
    mxGPUDestroyGPUArray(initDel);
    mxGPUDestroyGPUArray(elemFst);
    mxGPUDestroyGPUArray(elemLst);
    mxGPUDestroyGPUArray(rxElemOrig);
    mxGPUDestroyGPUArray(nSampOmit);
    mxGPUDestroyGPUArray(minRxTang);
    mxGPUDestroyGPUArray(maxRxTang);
    
    //cudaDeviceReset();
}
