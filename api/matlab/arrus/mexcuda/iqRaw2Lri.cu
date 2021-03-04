#define M_PI 3.14159265358979
#include "mex.h"
#include "gpu/mxGPUArray.h"


__constant__ float xElemConst[1024];


texture <float2, cudaTextureType1DLayered, cudaReadModeElementType> iqRawTex;


__forceinline__ __device__ float ownHypotf(float x, float y)
{
    return sqrtf(x*x + y*y);
}


__global__ void iqRaw2Lri(  float2 * iqLri, float const * zPix, float const * xPix, 
                            float const sos, float const fs, float const fn, 
                            float const txFoc, float const txAng, 
                            float const txApCent, float const initDel, 
                            float const minRxTang, float const maxRxTang, 
                            int const nSamp, int const nElem, 
                            int const nZPix, int const nXPix)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    float txDist, rxDist, txTang, rxTang, txApod, rxApod, time, iSamp;
    float modSin, modCos, sampRe, sampIm, pixRe = 0.f, pixIm = 0.f, pixWgh = 0.f;
    float const omega = 2 * M_PI * fn;
    float const sosInv = 1 / sos;
    float const zDistInv = 1 / zPix[z];
    float const rngRxTangInv = 2 / (maxRxTang - minRxTang); // inverted half range
    float const centRxTang = (maxRxTang + minRxTang) * 0.5f;
    float const nSigma = 2; // number of sigmas in half of the apodization Gaussian curve
    float const twoSigSqrInv = nSigma * nSigma * 0.5f;
    
    if (!isinf(txFoc)) {
        /* STA */
        float xFoc	= txFoc * sinf(txAng) + txApCent;
        float zFoc	= txFoc * cosf(txAng);
        float minTxTang = (xFoc - xElemConst[0      ]) / zFoc;
        float maxTxTang = (xFoc - xElemConst[nElem-1]) / zFoc;  // invalid tx aperture edges (temporary solution)!!!
        
        txDist	= ownHypotf(zPix[z] - zFoc, xPix[x] - xFoc);
        //txDist	= txDist * sign(zPix[z] - zFoc) + txFoc;          // WARNING: sign()=0 => invalid txDist value
        txTang	= (xPix[x] - xFoc) / (zPix[z] - zFoc);
        txApod	= (txTang >= minTxTang && txTang <= maxTxTang) ? 1.f : 0.f;
    }
    else {
        /* PWI */
        float r1 = (xPix[x]-xElemConst[0      ]) * cosf(txAng) - zPix[z] * sinf(txAng);
        float r2 = (xPix[x]-xElemConst[nElem-1]) * cosf(txAng) - zPix[z] * sinf(txAng);
        
        txDist = xPix[x] * sinf(txAng) + zPix[z] * cosf(txAng);
        txApod = (r1 >= 0.f && r2 <= 0.f) ? 1.f : 0.f;
    }
    
    for (int iElem=0; iElem<nElem; iElem++) {

        rxDist = ownHypotf(zPix[z], xPix[x] - xElemConst[iElem]);   // +10us
        rxTang = (xPix[x] - xElemConst[iElem]) * zDistInv;          // 4us
        if (rxTang < minRxTang || rxTang > maxRxTang) continue;
        rxApod = (rxTang-centRxTang)*rngRxTangInv;
        rxApod = __expf(-rxApod*rxApod*twoSigSqrInv);
        
        time = (txDist + rxDist) * sosInv + initDel;
        iSamp = time * fs;
        if (iSamp<0.f && iSamp>static_cast<float>(nSamp-1)) continue;
            
        float2 iqSamp = tex1DLayered(iqRawTex, iSamp + 0.5f, iElem);
        sampRe = iqSamp.x;
        sampIm = iqSamp.y;
            
        __sincosf(omega * time, &modSin, &modCos);
            
        pixRe += (sampRe * modCos - sampIm * modSin) * rxApod; // 60-80us
        pixIm += (sampRe * modSin + sampIm * modCos) * rxApod;
        pixWgh += rxApod;
    }
    
    iqLri[z + x*nZPix].x = pixRe / pixWgh * txApod;
    iqLri[z + x*nZPix].y = pixIm / pixWgh * txApod;
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
    
    float2 * dev_iqLri;
    void * dev_iqRaw;
    float const * dev_xElem;
    float const * dev_zPix;
    float const * dev_xPix;
    
    float sos;
    float fs;
    float fn;
    
    float foc;
    float ang;
    float cent;
    float initDel;
    float minRxTang;
    float maxRxTang;
    
    int nSamp;
    int nElem;
    int nZPix;
    int nXPix;
    
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
    
    for (int i=0; i<4; i++) {
        if (!(mxIsGPUArray(prhs[i]))) {
            mexErrMsgIdAndTxt( invalidInputMsgId, "First 4 inputs must be gpuArray");
        }
    }
    
    for (int i=4; i<13; i++) {
        if (!mxIsSingle(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
            mexErrMsgIdAndTxt( invalidInputMsgId, "Last 9 inputs must be single, real scalars");
        }
    }
    
    
    /* Extract inputs from prhs */
    iqRaw = mxGPUCreateFromMxArray(prhs[0]);
    xElem = mxGPUCreateFromMxArray(prhs[1]);
    zPix  = mxGPUCreateFromMxArray(prhs[2]);
    xPix  = mxGPUCreateFromMxArray(prhs[3]);
    
    sos   = mxGetScalar(prhs[4]);
    fs    = mxGetScalar(prhs[5]);
    fn    = mxGetScalar(prhs[6]);
    
    foc   = mxGetScalar(prhs[7]);
    ang   = mxGetScalar(prhs[8]);
    cent  = mxGetScalar(prhs[9]);
    initDel	= mxGetScalar(prhs[10]);
    minRxTang = mxGetScalar(prhs[11]);
    maxRxTang = mxGetScalar(prhs[12]);
    
    
    /* Validate inputs */
    if ((mxGPUGetClassID(iqRaw) != mxSINGLE_CLASS) || !mxGPUGetComplexity(iqRaw) || mxGPUGetNumberOfDimensions(iqRaw) > 2) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "iqRaw must be single, complex 2D array.");
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
    
    if (mxGPUGetDimensions(iqRaw)[1] != mxGPUGetNumberOfElements(xElem)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "size(iqRaw,2) must be equal to length(xElem).");
    }
    
    /* Get some additional information */
    nSamp = mxGPUGetDimensions(iqRaw)[0];
    nElem = mxGPUGetNumberOfElements(xElem);
    nZPix = mxGPUGetNumberOfElements(zPix);
    nXPix = mxGPUGetNumberOfElements(xPix);
    
    sharedPerBlock = 0;
    blocksPerGrid = {(unsigned int)ceilf(static_cast<float>(nZPix)/static_cast<float>(threadsPerBlock.x)), 
                     (unsigned int)ceilf(static_cast<float>(nXPix)/static_cast<float>(threadsPerBlock.y)), 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 2;
    mwSize dimOut[2] = {nZPix, nXPix};
    
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
    cudaExtent cuArraySize =  make_cudaExtent(nSamp, 0, nElem);
    cudaMalloc3DArray(&cuArray, &iqRawTex.channelDesc, cuArraySize, cudaArrayLayered);
    cudaMemcpy3DParms cuArrayCopy = {0};
    cuArrayCopy.srcPtr = make_cudaPitchedPtr(dev_iqRaw, nSamp * sizeof(float2), nSamp, 1);
    cuArrayCopy.dstArray = cuArray;
    cuArrayCopy.extent = make_cudaExtent(nSamp, 1, nElem);
    cuArrayCopy.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&cuArrayCopy);

    cudaBindTextureToArray(iqRawTex, cuArray);
    
    /* Execute CUDA kernel */
    iqRaw2Lri<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqLri, dev_zPix, dev_xPix, 
                                                                  sos, fs, fn, foc, ang, cent, initDel, minRxTang, maxRxTang, 
                                                                  nSamp, nElem, nZPix, nXPix);
    
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
    
    //cudaDeviceReset();
}
