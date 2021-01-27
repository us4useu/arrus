#define _USE_MATH_DEFINES
#include "math.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"

__global__ void iqRaw2Lri(  float2 * iqLri, float2 const * iqRaw, 
                            float const * xElem, float const * zPix, float const * xPix, 
                            float const sos, float const fs, float const fn, 
                            int const nSamp, int const nElem, 
                            int const nZPix, int const nXPix)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    float txDist, rxDist, time, iSamp, apodWgh, interpWgh;
    float modSin, modCos, sampRe, sampIm, pixRe = 0.f, pixIm = 0.f, pixWgh = 0.f;
    int offset;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    txDist = zPix[z];
    
    for (int iElem=0; iElem<nElem; iElem++) {
        
        rxDist = hypotf(zPix[z], xPix[x] - xElem[iElem]);
        time = (txDist + rxDist) / sos;
        iSamp = time * fs;
        apodWgh = (fabsf(xPix[x] - xElem[iElem]) / zPix[z] <= 0.5f) ? 1.f : 0.f;
        
        if (iSamp>=0.f && iSamp<=(float)nSamp) {
            offset = iElem * nSamp;
            interpWgh = 1 - (iSamp - floorf(iSamp));
            
            modSin = sinf(2 * M_PI * fn * time);
            modCos = cosf(2 * M_PI * fn * time);
            
            sampRe = (iqRaw[offset + (int)floorf(iSamp)].x * interpWgh
                    + iqRaw[offset + (int) ceilf(iSamp)].x * (1 - interpWgh));
            sampIm = (iqRaw[offset + (int)floorf(iSamp)].y * interpWgh
                    + iqRaw[offset + (int) ceilf(iSamp)].y * (1 - interpWgh));
            
            pixRe += (sampRe * modCos - sampIm * modSin) * apodWgh;
            pixIm += (sampRe * modSin + sampIm * modCos) * apodWgh;
            pixWgh += apodWgh;
        }
    }
    
    iqLri[z + x*nZPix].x = pixRe / pixWgh;
    iqLri[z + x*nZPix].y = pixIm / pixWgh;
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
    float2 const * dev_iqRaw;
    float const * dev_xElem;
    float const * dev_zPix;
    float const * dev_xPix;
    
    float sos;
    float fs;
    float fn;
    int nSamp;
    int nElem;
    int nZPix;
    int nXPix;
    
    dim3 const threadsPerBlock = {32, 32, 1};
    dim3 blocksPerGrid;
    
    /* Validate mex inputs/outputs */
    if (nrhs!=7) {
        mexErrMsgIdAndTxt("testRecMex:InvalidInput", "Seven inputs required");
    }
    
    if (nlhs!=1) {
        mexErrMsgIdAndTxt("testRecMex:InvalidOutput", "One output allowed");
    }
    
    if (!(mxIsGPUArray(prhs[0])) || 
        !(mxIsGPUArray(prhs[1])) || 
        !(mxIsGPUArray(prhs[2])) || 
        !(mxIsGPUArray(prhs[3]))) {
        mexErrMsgIdAndTxt("testRecMex:InvalidInput", "First 4 inputs (iqRaw, xElem, zPix, xPix) must be gpuArray");
    }
    
    if (!mxIsSingle(prhs[4]) || !mxIsSingle(prhs[5]) || !mxIsSingle(prhs[6]) || 
        mxIsComplex(prhs[4]) || mxIsComplex(prhs[5]) || mxIsComplex(prhs[6]) || 
        mxGetNumberOfElements(prhs[4]) != 1 || mxGetNumberOfElements(prhs[5]) != 1 || mxGetNumberOfElements(prhs[6]) != 1) {
        mexErrMsgIdAndTxt("testRecMex:InvalidInput", "Last 3 inputs (sos, fs, fn) must be single, real scalars");
    }
    
    /* Extract inputs from prhs */
    iqRaw = mxGPUCreateFromMxArray(prhs[0]);
    xElem = mxGPUCreateFromMxArray(prhs[1]);
    zPix  = mxGPUCreateFromMxArray(prhs[2]);
    xPix  = mxGPUCreateFromMxArray(prhs[3]);
    
    sos   = mxGetScalar(prhs[4]);
    fs    = mxGetScalar(prhs[5]);
    fn    = mxGetScalar(prhs[6]);
    
    /* Validate inputs */
    if ((mxGPUGetClassID(iqRaw) != mxSINGLE_CLASS) || !mxGPUGetComplexity(iqRaw) || mxGPUGetNumberOfDimensions(iqRaw) > 2) {
        mexErrMsgIdAndTxt("testRecMex:InvalidInput", "iqRaw must be single, complex 2D array.");
    }
    
    if ((mxGPUGetClassID(xElem) != mxSINGLE_CLASS) || mxGPUGetComplexity(xElem) || !(mxGPUGetNumberOfDimensions(xElem) == 2 && mxGPUGetDimensions(xElem)[0] == 1)) {
        mexErrMsgIdAndTxt("testRecMex:InvalidInput", "xElem must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(zPix) != mxSINGLE_CLASS) || mxGPUGetComplexity(zPix) || !(mxGPUGetNumberOfDimensions(zPix) == 2 && mxGPUGetDimensions(zPix)[0] == 1)) {
        mexErrMsgIdAndTxt("testRecMex:InvalidInput", "zPix must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(xPix) != mxSINGLE_CLASS) || mxGPUGetComplexity(xPix) || !(mxGPUGetNumberOfDimensions(xPix) == 2 && mxGPUGetDimensions(xPix)[0] == 1)) {
        mexErrMsgIdAndTxt("testRecMex:InvalidInput", "xPix must be single, real, horizontal vector.");
    }
    
    if (mxGPUGetDimensions(iqRaw)[1] != mxGPUGetNumberOfElements(xElem)) {
        mexErrMsgIdAndTxt("testRecMex:InvalidInput", "size(iqRaw,2) must be equal to length(xElem).");
    }
    
    /* Get some additional information */
    nSamp = mxGPUGetDimensions(iqRaw)[0];
    nElem = mxGPUGetNumberOfElements(xElem);
    nZPix = mxGPUGetNumberOfElements(zPix);
    nXPix = mxGPUGetNumberOfElements(xPix);
    
    blocksPerGrid = {(unsigned int)ceilf((float)nZPix/(float)threadsPerBlock.x), 
                     (unsigned int)ceilf((float)nXPix/(float)threadsPerBlock.y), 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 2;
    mwSize dimOut[2] = {nZPix, nXPix};
    
    iqLri = mxGPUCreateGPUArray(nDimOut,
                                dimOut,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_iqLri = (float2 *)(mxGPUGetData(iqLri));
    dev_iqRaw = (float2 const *)(mxGPUGetDataReadOnly(iqRaw));
    dev_xElem = (float const *)(mxGPUGetDataReadOnly(xElem));
    dev_zPix  = (float const *)(mxGPUGetDataReadOnly(zPix));
    dev_xPix  = (float const *)(mxGPUGetDataReadOnly(xPix));
    
    /* Execute CUDA kernel */
    iqRaw2Lri<<<blocksPerGrid, threadsPerBlock>>>(dev_iqLri, dev_iqRaw, dev_xElem, dev_zPix, dev_xPix, 
                                                    sos, fs, fn, nSamp, nElem, nZPix, nXPix);
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(iqLri);
    
    /* Destroy the mxGPUArray objects */
    mxGPUDestroyGPUArray(iqLri);
    mxGPUDestroyGPUArray(iqRaw);
    mxGPUDestroyGPUArray(xElem);
    mxGPUDestroyGPUArray(zPix);
    mxGPUDestroyGPUArray(xPix);
    
    //cudaDeviceReset();
}
