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


__global__ void iqRaw2Lin(  float2 * iqLin, 
                            float const * rPix, 
                            float const * txAngZX, 
                            float const * txApCentZ, 
                            float const * txApCentX, 
                            float const * fn, 
                            float const * initDel, 
                            int const * rxApOrigElem, 
                            int const * nSampOmit, 
                            float const * minRxTang, 
                            float const * maxRxTang, 
                            float const fs, 
                            float const sos, 
                            int const nPix, 
                            int const nSamp, 
                            int const nElem, 
                            int const nRx, 
                            int const nTx)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int iTx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r>=nPix || iTx>=nTx) {
        return;
    }
    
    int iElem;
    float rxDist, rxTang, rxApod, time, iSamp;
    float modSin, modCos, sampRe, sampIm, pixRe, pixIm, pixWgh;
    float const sosInv = 1 / sos;
    
    float zPix = txApCentZ[iTx] + rPix[r] * cosf(txAngZX[iTx]);
    float xPix = txApCentX[iTx] + rPix[r] * sinf(txAngZX[iTx]);
    
    iElem = (rxApOrigElem[iTx] + nRx <= nElem) ? rxApOrigElem[iTx] + nRx - 1 : nElem - 1;
    rxTang = __fdividef(xPix - xElemConst[iElem], zPix - zElemConst[iElem]);
    rxTang = __fdividef(rxTang - tangElemConst[iElem], 1.f + rxTang*tangElemConst[iElem]);
    float minRxTangPix = fmax(minRxTang[iTx], rxTang);
    
    iElem = (rxApOrigElem[iTx] >= 0) ? rxApOrigElem[iTx] : 0;
    rxTang = __fdividef(xPix - xElemConst[iElem], zPix - zElemConst[iElem]);
    rxTang = __fdividef(rxTang - tangElemConst[iElem], 1.f + rxTang*tangElemConst[iElem]);
    float maxRxTangPix = fmin(maxRxTang[iTx], rxTang);
    
    float const rngRxTangInv = 1 / (maxRxTangPix - minRxTangPix); // inverted tangent range
    float omega = 2 * M_PI * fn[iTx];
    
    pixRe = 0.f;
    pixIm = 0.f;
    pixWgh = 0.f;
    
    for (int iRx=0; iRx<nRx; iRx++) {
        iElem = iRx + rxApOrigElem[iTx];
        if (iElem<0 || iElem>=nElem) continue;
        
        rxDist = ownHypotf(xPix - xElemConst[iElem], zPix - zElemConst[iElem]);
        rxTang = __fdividef(xPix - xElemConst[iElem], zPix - zElemConst[iElem]);
        rxTang = __fdividef(rxTang-tangElemConst[iElem], 1.f+rxTang*tangElemConst[iElem]);
        if (rxTang < minRxTangPix || rxTang > maxRxTangPix) continue;
        rxApod = (rxTang-minRxTangPix)*rngRxTangInv; // <0,1>, needs normalized texture fetching, errors at aperture sided
        rxApod = tex1D(rxApodTex, rxApod);
        
        time = (rPix[r] + rxDist) * sosInv + initDel[iTx];
        iSamp = time * fs;
        if (iSamp<static_cast<float>(nSampOmit[iTx]) || iSamp>static_cast<float>(nSamp-1)) continue;
        
        float2 iqSamp = tex1DLayered(iqRawTex, iSamp + 0.5f, iRx + iTx*nRx);
        sampRe = iqSamp.x;
        sampIm = iqSamp.y;
        
        __sincosf(omega * time, &modSin, &modCos);
        
        pixRe += (sampRe * modCos - sampIm * modSin) * rxApod;
        pixIm += (sampRe * modSin + sampIm * modCos) * rxApod;
        pixWgh += rxApod;
    }
    
    iqLin[r + iTx*nPix].x = pixRe / pixWgh;
    iqLin[r + iTx*nPix].y = pixIm / pixWgh;
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
    mxGPUArray * iqLin;
    mxGPUArray const * iqRaw;
    mxGPUArray const * zElem;
    mxGPUArray const * xElem;
    mxGPUArray const * tangElem;
    mxGPUArray const * rPix;
    mxGPUArray const * rxApod;
    mxGPUArray const * ang;
    mxGPUArray const * centZ;
    mxGPUArray const * centX;
    mxGPUArray const * fn;
    mxGPUArray const * initDel;
    mxGPUArray const * rxElemOrig;
    mxGPUArray const * nSampOmit;
    mxGPUArray const * minRxTang;
    mxGPUArray const * maxRxTang;
    
    float2 * dev_iqLin;
    float2 const * dev_iqRaw;
    float const * dev_zElem;
    float const * dev_xElem;
    float const * dev_tangElem;
    float const * dev_rPix;
    float const * dev_rxApod;
    float const * dev_ang;
    float const * dev_centZ;
    float const * dev_centX;
    float const * dev_fn;
    float const * dev_initDel;
    int const * dev_rxElemOrig;
    int const * dev_nSampOmit;
    float const * dev_minRxTang;
    float const * dev_maxRxTang;
    
    float fs;
    float sos;
    
    int nSamp;
    int nElem;
    int nPix;
    int nRx;
    int nTx;
    int nRxApodSamp;
    
    dim3 const threadsPerBlock = {32, 16, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
    char const * const invalidInputMsgId = "iqRaw2Lin:InvalidInput";
    char const * const invalidOutputMsgId = "iqRaw2Lin:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=17) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "17 inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }
    
    /* Extract inputs from prhs */
    iqRaw     = mxGPUCreateFromMxArray(prhs[0]);
    zElem     = mxGPUCreateFromMxArray(prhs[1]);
    xElem     = mxGPUCreateFromMxArray(prhs[2]);
    tangElem  = mxGPUCreateFromMxArray(prhs[3]);
    rPix      = mxGPUCreateFromMxArray(prhs[4]);
    rxApod    = mxGPUCreateFromMxArray(prhs[5]);
    ang       = mxGPUCreateFromMxArray(prhs[6]);
    centZ     = mxGPUCreateFromMxArray(prhs[7]);
    centX     = mxGPUCreateFromMxArray(prhs[8]);
    fn        = mxGPUCreateFromMxArray(prhs[9]);
    initDel   = mxGPUCreateFromMxArray(prhs[10]);
    rxElemOrig= mxGPUCreateFromMxArray(prhs[11]);
    nSampOmit = mxGPUCreateFromMxArray(prhs[12]);
    minRxTang = mxGPUCreateFromMxArray(prhs[13]);
    maxRxTang = mxGPUCreateFromMxArray(prhs[14]);

    fs        = mxGetScalar(prhs[15]);
    sos       = mxGetScalar(prhs[16]);
    
    /* Validate inputs */
    checkData(iqRaw,     "iqRaw",     false, true,  3, invalidInputMsgId);
    checkData(zElem,     "zElem",     false, false, 1, invalidInputMsgId);
    checkData(xElem,     "xElem",     false, false, 1, invalidInputMsgId);
    checkData(tangElem,  "tangElem",  false, false, 1, invalidInputMsgId);
    checkData(rPix,      "rPix",      false, false, 1, invalidInputMsgId);
    checkData(rxApod,    "rxApod",    false, false, 1, invalidInputMsgId);
    checkData(ang,       "ang",       false, false, 1, invalidInputMsgId);
    checkData(centZ,     "centZ",     false, false, 1, invalidInputMsgId);
    checkData(centX,     "centX",     false, false, 1, invalidInputMsgId);
    checkData(fn,        "fn",        false, false, 1, invalidInputMsgId);
    checkData(initDel,   "initDel",   false, false, 1, invalidInputMsgId);
    checkData(rxElemOrig,"rxElemOrig",true,  false, 1, invalidInputMsgId);
    checkData(nSampOmit, "nSampOmit", true,  false, 1, invalidInputMsgId);
    checkData(minRxTang, "minRxTang", false, false, 1, invalidInputMsgId);
    checkData(maxRxTang, "maxRxTang", false, false, 1, invalidInputMsgId);
    
    /* Get some additional information */
    nSamp = mxGPUGetDimensions(iqRaw)[0];
    nRx   = mxGPUGetDimensions(iqRaw)[1];
    nElem = mxGPUGetNumberOfElements(xElem);
    nPix  = mxGPUGetNumberOfElements(rPix);
    nRxApodSamp = mxGPUGetNumberOfElements(rxApod);
    if (mxGPUGetNumberOfDimensions(iqRaw)<3) {
        nTx = 1;
    }
    else {
        nTx   = mxGPUGetDimensions(iqRaw)[2];
    }
    
    sharedPerBlock = 0;
    blocksPerGrid = {(nPix+threadsPerBlock.x-1)/threadsPerBlock.x, 
                     (nTx+threadsPerBlock.y-1)/threadsPerBlock.y, 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 2;
    mwSize dimOut[2] = {nPix, nTx};
    
    iqLin = mxGPUCreateGPUArray(nDimOut,
                                dimOut,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_iqLin    = static_cast<float2 *>(mxGPUGetData(iqLin));
    dev_iqRaw    = static_cast<float2 const *>(mxGPUGetDataReadOnly(iqRaw));
    dev_zElem    = static_cast<float const *>(mxGPUGetDataReadOnly(zElem));
    dev_xElem    = static_cast<float const *>(mxGPUGetDataReadOnly(xElem));
    dev_tangElem = static_cast<float const *>(mxGPUGetDataReadOnly(tangElem));
    dev_rPix     = static_cast<float const *>(mxGPUGetDataReadOnly(rPix));
    dev_rxApod   = static_cast<float const *>(mxGPUGetDataReadOnly(rxApod));
    dev_ang      = static_cast<float const *>(mxGPUGetDataReadOnly(ang));
    dev_centZ    = static_cast<float const *>(mxGPUGetDataReadOnly(centZ));
    dev_centX    = static_cast<float const *>(mxGPUGetDataReadOnly(centX));
    dev_fn       = static_cast<float const *>(mxGPUGetDataReadOnly(fn));
    dev_initDel  = static_cast<float const *>(mxGPUGetDataReadOnly(initDel));
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
        iqRaw2Lin<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqLin + iPart*nPix*nTxPerPart, 
                                                                      dev_rPix, 
                                                                      dev_ang       + iPart*nTxPerPart, 
                                                                      dev_centZ     + iPart*nTxPerPart, 
                                                                      dev_centX     + iPart*nTxPerPart, 
                                                                      dev_fn        + iPart*nTxPerPart, 
                                                                      dev_initDel   + iPart*nTxPerPart, 
                                                                      dev_rxElemOrig + iPart*nTxPerPart, 
                                                                      dev_nSampOmit + iPart*nTxPerPart, 
                                                                      dev_minRxTang + iPart*nTxPerPart, 
                                                                      dev_maxRxTang + iPart*nTxPerPart, 
                                                                      fs, sos, 
                                                                      nPix, nSamp, nElem, nRx, nTxInThisPart);
    }
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(iqLin);
    
    /* Clean-up */
    cudaUnbindTexture(iqRawTex);
    cudaFreeArray(cuArray);
    
    cudaUnbindTexture(rxApodTex);
    cudaFreeArray(cuArrayApod);
    
    mxGPUDestroyGPUArray(iqLin);
    mxGPUDestroyGPUArray(iqRaw);
    mxGPUDestroyGPUArray(zElem);
    mxGPUDestroyGPUArray(xElem);
    mxGPUDestroyGPUArray(tangElem);
    mxGPUDestroyGPUArray(rPix);
    mxGPUDestroyGPUArray(rxApod);
    mxGPUDestroyGPUArray(ang);
    mxGPUDestroyGPUArray(centZ);
    mxGPUDestroyGPUArray(centX);
    mxGPUDestroyGPUArray(fn);
    mxGPUDestroyGPUArray(initDel);
    mxGPUDestroyGPUArray(rxElemOrig);
    mxGPUDestroyGPUArray(nSampOmit);
    mxGPUDestroyGPUArray(minRxTang);
    mxGPUDestroyGPUArray(maxRxTang);
    
    //cudaDeviceReset();
}
