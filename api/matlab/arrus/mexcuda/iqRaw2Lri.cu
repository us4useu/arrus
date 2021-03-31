#define M_PI 3.14159265358979
#include "mex.h"
#include "gpu/mxGPUArray.h"
// #include <string>

__constant__ float xElemConst[1024];


texture <float2, cudaTextureType1DLayered, cudaReadModeElementType> iqRawTex;


__forceinline__ __device__ float ownHypotf(float x, float y)
{
    return sqrtf(x*x + y*y);
}


__global__ void iqRaw2Lri(  float2 * iqLri, float const * zPix, float const * xPix, 
                            float const * txFoc, float const * txAng, float const * txApCent, 
                            float const * minRxTang, float const * maxRxTang, 
                            float const fs, float const fn, 
                            float const sos, float const initDel, 
                            int const nZPix, int const nXPix, 
                            int const nSamp, int const nElem, 
                            int const nTx, int const nRepTx, int const nRepSeq)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    float txDist, rxDist, txTang, rxTang, txApod, rxApod, time, iSamp;
    float modSin, modCos, sampRe, sampIm, pixRe, pixIm, pixWgh;
    float const omega = 2 * M_PI * fn;
    float const sosInv = 1 / sos;
    float const zDistInv = 1 / zPix[z];
    float const nSigma = 3; // number of sigmas in half of the apodization Gaussian curve
    float const twoSigSqrInv = nSigma * nSigma * 0.5f;
    
    for (int iTx=0; iTx<nTx; iTx++) {
        
        if (!isinf(txFoc[iTx])) {
            /* STA */
            float xFoc	= txFoc[iTx] * sinf(txAng[iTx]) + txApCent[iTx];
            float zFoc	= txFoc[iTx] * cosf(txAng[iTx]);
            float minTxTang = (xFoc - xElemConst[0      ]) / zFoc;
            float maxTxTang = (xFoc - xElemConst[nElem-1]) / zFoc;  // invalid tx aperture edges (temporary solution)!!!
            
            txDist	= ownHypotf(zPix[z] - zFoc, xPix[x] - xFoc);
            //txDist	= txDist * sign(zPix[z] - zFoc) + txFoc[iTx];          // WARNING: sign()=0 => invalid txDist value
            txTang	= (xPix[x] - xFoc) / (zPix[z] - zFoc);
            txApod	= (txTang >= minTxTang && txTang <= maxTxTang) ? 1.f : 0.f;
        }
        else {
            /* PWI */
            float r1 = (xPix[x]-xElemConst[0      ]) * cosf(txAng[iTx]) - zPix[z] * sinf(txAng[iTx]);
            float r2 = (xPix[x]-xElemConst[nElem-1]) * cosf(txAng[iTx]) - zPix[z] * sinf(txAng[iTx]);
            
            txDist = xPix[x] * sinf(txAng[iTx]) + zPix[z] * cosf(txAng[iTx]);
            txApod = (r1 >= 0.f && r2 <= 0.f) ? 1.f : 0.f;
        }
        
        float rngRxTangInv = 2 / (maxRxTang[iTx] - minRxTang[iTx]); // inverted half range
        float centRxTang = (maxRxTang[iTx] + minRxTang[iTx]) * 0.5f;
        
        pixRe = 0.f;
        pixIm = 0.f;
        pixWgh = 0.f;
        
        for (int iElem=0; iElem<nElem; iElem++) {
            rxDist = ownHypotf(zPix[z], xPix[x] - xElemConst[iElem]);   // +10us
            rxTang = (xPix[x] - xElemConst[iElem]) * zDistInv;          // 4us
            if (rxTang < minRxTang[iTx] || rxTang > maxRxTang[iTx]) continue;
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
        
        iqLri[z + x*nZPix + iTx*nZPix*nXPix].x = pixRe / pixWgh * txApod;
        iqLri[z + x*nZPix + iTx*nZPix*nXPix].y = pixIm / pixWgh * txApod;
    }
}

// __host__ void checkData(void const * const data, std::string const name, bool const isComplex, int const nDims)
// {
//     std::string const invalidInputMsgId = "iqRaw2Lri:InvalidInput";
//     
//     if (mxGPUGetClassID(data) != mxSINGLE_CLASS) 
//         mexErrMsgIdAndTxt( invalidInputMsgId, name + " must be single.");
//     
//     else if (!isComplex && mxGPUGetComplexity(data)) 
//         mexErrMsgIdAndTxt( invalidInputMsgId, name + " must be real.");
//     
//     else if (isComplex && !mxGPUGetComplexity(data)) 
//         mexErrMsgIdAndTxt( invalidInputMsgId, name + " must be complex.");
//     
//     else if (nDims==1 && !( mxGPUGetNumberOfDimensions(data) == 1 || 
//                            (mxGPUGetNumberOfDimensions(data) == 2 && mxGPUGetDimensions(data)[0] == 1))) 
//         mexErrMsgIdAndTxt( invalidInputMsgId, name + " must be 1D vector.");
//     
//     else if (nDims==2 && !(mxGPUGetNumberOfDimensions(data) == 2)) 
//         mexErrMsgIdAndTxt( invalidInputMsgId, name + " must be 2D array.");
//     
//     else if (nDims==3 && !(mxGPUGetNumberOfDimensions(data) == 3)) 
//         mexErrMsgIdAndTxt( invalidInputMsgId, name + " must be 3D array.");
//     
// }

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
    mxGPUArray const * minRxTang;
    mxGPUArray const * maxRxTang;
    
    float2 * dev_iqLri;
    void * dev_iqRaw;
    float const * dev_xElem;
    float const * dev_zPix;
    float const * dev_xPix;
    
    float const * dev_foc;
    float const * dev_ang;
    float const * dev_cent;
    float const * dev_minRxTang;
    float const * dev_maxRxTang;
    
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
    
    for (int i=9; i<13; i++) {
        if (!mxIsSingle(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
            mexErrMsgIdAndTxt( invalidInputMsgId, "Last 4 inputs must be single, real scalars");
        }
    }
    
    
    /* Extract inputs from prhs */
    iqRaw = mxGPUCreateFromMxArray(prhs[0]);
    xElem = mxGPUCreateFromMxArray(prhs[1]);
    zPix  = mxGPUCreateFromMxArray(prhs[2]);
    xPix  = mxGPUCreateFromMxArray(prhs[3]);
    
    foc   = mxGPUCreateFromMxArray(prhs[4]);
    ang   = mxGPUCreateFromMxArray(prhs[5]);
    cent  = mxGPUCreateFromMxArray(prhs[6]);
    minRxTang = mxGPUCreateFromMxArray(prhs[7]);
    maxRxTang = mxGPUCreateFromMxArray(prhs[8]);
    
    fs    = mxGetScalar(prhs[9]);
    fn    = mxGetScalar(prhs[10]);
    sos   = mxGetScalar(prhs[11]);
    initDel	= mxGetScalar(prhs[12]);
    
    /* Validate inputs */
    if ((mxGPUGetClassID(iqRaw) != mxSINGLE_CLASS) || !mxGPUGetComplexity(iqRaw) || mxGPUGetNumberOfDimensions(iqRaw) > 3) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "iqRaw must be single, complex 3D array.");
    }
    
    if ((mxGPUGetClassID(xElem) != mxSINGLE_CLASS) || mxGPUGetComplexity(xElem) || !(mxGPUGetNumberOfDimensions(xElem) == 2 && mxGPUGetDimensions(xElem)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "xElem must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(zPix) != mxSINGLE_CLASS) || mxGPUGetComplexity(zPix) || !(mxGPUGetNumberOfDimensions(zPix) == 2 && mxGPUGetDimensions(zPix)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "zPix must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(xPix) != mxSINGLE_CLASS) || mxGPUGetComplexity(xPix) || !(mxGPUGetNumberOfDimensions(xPix) == 2 && mxGPUGetDimensions(xPix)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "xPix must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(foc) != mxSINGLE_CLASS) || mxGPUGetComplexity(foc) || !(mxGPUGetNumberOfDimensions(foc) == 2 && mxGPUGetDimensions(foc)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "foc must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(ang) != mxSINGLE_CLASS) || mxGPUGetComplexity(ang) || !(mxGPUGetNumberOfDimensions(ang) == 2 && mxGPUGetDimensions(ang)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "ang must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(cent) != mxSINGLE_CLASS) || mxGPUGetComplexity(cent) || !(mxGPUGetNumberOfDimensions(cent) == 2 && mxGPUGetDimensions(cent)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "cent must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(minRxTang) != mxSINGLE_CLASS) || mxGPUGetComplexity(minRxTang) || !(mxGPUGetNumberOfDimensions(minRxTang) == 2 && mxGPUGetDimensions(minRxTang)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "minRxTang must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(maxRxTang) != mxSINGLE_CLASS) || mxGPUGetComplexity(maxRxTang) || !(mxGPUGetNumberOfDimensions(maxRxTang) == 2 && mxGPUGetDimensions(maxRxTang)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "maxRxTang must be single, real, horizontal vector.");
    }
    
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
    blocksPerGrid = {(unsigned int)ceilf(static_cast<float>(nZPix)/static_cast<float>(threadsPerBlock.x)), 
                     (unsigned int)ceilf(static_cast<float>(nXPix)/static_cast<float>(threadsPerBlock.y)), 1};
    
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
    dev_iqRaw = const_cast<void *>(mxGPUGetDataReadOnly(iqRaw));
    dev_xElem = static_cast<float const *>(mxGPUGetDataReadOnly(xElem));
    dev_zPix  = static_cast<float const *>(mxGPUGetDataReadOnly(zPix));
    dev_xPix  = static_cast<float const *>(mxGPUGetDataReadOnly(xPix));
    
    dev_foc   = static_cast<float const *>(mxGPUGetDataReadOnly(foc));
    dev_ang   = static_cast<float const *>(mxGPUGetDataReadOnly(ang));
    dev_cent  = static_cast<float const *>(mxGPUGetDataReadOnly(cent));
    dev_minRxTang  = static_cast<float const *>(mxGPUGetDataReadOnly(minRxTang));
    dev_maxRxTang  = static_cast<float const *>(mxGPUGetDataReadOnly(maxRxTang));
    
    /* set constant memory */
    if(nElem > 1024) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "xElem is too long, kernel supports xElem of up to 1024 elements");
    }
    cudaMemcpyToSymbol(xElemConst, dev_xElem, nElem*4, 0, cudaMemcpyDeviceToDevice);
    
    /* configure texture reference */
    iqRawTex.normalized  = false;
    iqRawTex.addressMode[0] = cudaAddressModeBorder;
    iqRawTex.filterMode  = cudaFilterModeLinear;
    
    cudaArray* cuArray;
    cudaExtent cuArraySize =  make_cudaExtent(nSamp, 0, nElem*nTx);
    cudaMalloc3DArray(&cuArray, &iqRawTex.channelDesc, cuArraySize, cudaArrayLayered);
    cudaMemcpy3DParms cuArrayCopy = {0};
    cuArrayCopy.srcPtr = make_cudaPitchedPtr(dev_iqRaw, nSamp * sizeof(float2), nSamp, 1);
    cuArrayCopy.dstArray = cuArray;
    cuArrayCopy.extent = make_cudaExtent(nSamp, 1, nElem*nTx);
    cuArrayCopy.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&cuArrayCopy);

    cudaBindTextureToArray(iqRawTex, cuArray);
    
    /* Execute CUDA kernel */
    iqRaw2Lri<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqLri, dev_zPix, dev_xPix, 
                                                                  dev_foc, dev_ang, dev_cent, 
                                                                  dev_minRxTang, dev_maxRxTang, 
                                                                  fs, fn, sos, initDel, 
                                                                  nZPix, nXPix, nSamp, nElem, nTx, 1, 1);
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(iqLri);
    
    /* Destroy the mxGPUArray objects */
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
    mxGPUDestroyGPUArray(minRxTang);
    mxGPUDestroyGPUArray(maxRxTang);
    
    //cudaDeviceReset();
}
