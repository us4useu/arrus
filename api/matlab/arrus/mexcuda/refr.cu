#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <iostream>

__constant__ float zElemConst[256];
__constant__ float xElemConst[256];

__forceinline__ __device__ float ownHypotf(float x, float y)
{
    return sqrtf(x*x + y*y);
}

// __fdividef

__global__ void refr(       float * xRefract, 
                            int * nIter, 
                            float const * zPix, 
                            float const * xPix, 
                            float const sosInterf, 
                            float const sosSample,
                            float const timePrec, 
                            int const nZPix, 
                            int const nXPix, 
                            int const nElem)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    float xRefractLo, sinRatioLo, xRefractHi, sinRatioHi, timeOld;
    float xRefractNew, distInterf, distSample, sinRatioNew, timeNew;
	
    float const cRatio = sosInterf / sosSample;

    for (int iElem=0; iElem<nElem; iElem++) {
        
        // Initial refraction points
        xRefractLo = xElemConst[iElem];
        sinRatioLo = 0.f;
        
        xRefractHi = xElemConst[iElem] + (xPix[x] - xElemConst[iElem]) * -zElemConst[iElem] / (zPix[z] - zElemConst[iElem]);
        sinRatioHi = 1.f;
        
        timeOld = ownHypotf(xRefractHi - xElemConst[iElem], zElemConst[iElem]) / sosInterf
                + ownHypotf(xPix[x]    - xRefractHi,        zPix[z])           / sosSample;

        // Iterations
        int iIter = 0;
        do {
            xRefractNew = xRefractLo + (xRefractHi-xRefractLo)*(cRatio-sinRatioLo)/(sinRatioHi-sinRatioLo);
            distInterf  = ownHypotf(xRefractNew - xElemConst[iElem], zElemConst[iElem]);
            distSample  = ownHypotf(xPix[x]     - xRefractNew,       zPix[z]);
            sinRatioNew = ((xRefractNew - xElemConst[iElem]) / distInterf) 
                        / ((xPix[x]     - xRefractNew)       / distSample);
            timeNew     = distInterf / sosInterf + distSample / sosSample;

            if (sinRatioNew < cRatio) {
                xRefractLo = xRefractNew;
                sinRatioLo = sinRatioNew;
            }
            else {
                xRefractHi = xRefractNew;
                sinRatioHi = sinRatioNew;
            }
            timeOld = timeNew;
            iIter++;
        } while(fabs(timeNew-timeOld) > timePrec);
        
        xRefract[z + x*nZPix + iElem*nZPix*nXPix] = xRefractNew;
        nIter[z + x*nZPix + iElem*nZPix*nXPix] = iIter;
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
    mxGPUArray * xRefract;
    mxGPUArray * nIter;
    mxGPUArray const * zElem;
    mxGPUArray const * xElem;
    mxGPUArray const * zPix;
    mxGPUArray const * xPix;
    
    float * dev_xRefract;
    int * dev_nIter;
    float const * dev_zElem;
    float const * dev_xElem;
    float const * dev_zPix;
    float const * dev_xPix;
    
    float sosInterf;
    float sosSample;
    float timePrec;
    
    int nElem;
    int nZPix;
    int nXPix;
    
    dim3 const threadsPerBlock = {16, 16, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
            char const * const invalidInputMsgId = "refr:InvalidInput";
            char const * const invalidOutputMsgId = "refr:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=7) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "7 inputs required");
    }
    
    if (nlhs>2) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "2 outputs allowed");
    }
    
//     for (int i=13; i<19; i++) {
//         if (!mxIsSingle(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
//             mexErrMsgIdAndTxt( invalidInputMsgId, "Last 6 inputs must be single, real scalars");
//         }
//     }
    
    
    /* Extract inputs from prhs */
    zElem     = mxGPUCreateFromMxArray(prhs[0]);
    xElem     = mxGPUCreateFromMxArray(prhs[1]);
    zPix      = mxGPUCreateFromMxArray(prhs[2]);
    xPix      = mxGPUCreateFromMxArray(prhs[3]);
    
    sosInterf = mxGetScalar(prhs[4]);
    sosSample = mxGetScalar(prhs[5]);
    timePrec  = mxGetScalar(prhs[6]);
    
    /* Validate inputs */
    checkData(zElem,     "zElem",     false, 1, invalidInputMsgId);
    checkData(xElem,     "xElem",     false, 1, invalidInputMsgId);
    checkData(zPix,      "zPix",      false, 1, invalidInputMsgId);
    checkData(xPix,      "xPix",      false, 1, invalidInputMsgId);
    
    /* Get some additional information */
    nElem = mxGPUGetNumberOfElements(xElem);
    nZPix = mxGPUGetNumberOfElements(zPix);
    nXPix = mxGPUGetNumberOfElements(xPix);
    
    sharedPerBlock = 0;
    blocksPerGrid = {(nZPix+threadsPerBlock.x-1)/threadsPerBlock.x, 
                     (nXPix+threadsPerBlock.y-1)/threadsPerBlock.y, 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 3;
    mwSize dimOut[3] = {nZPix, nXPix, nElem};
    
    xRefract = mxGPUCreateGPUArray(nDimOut, dimOut, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    nIter    = mxGPUCreateGPUArray(nDimOut, dimOut, mxINT16_CLASS,  mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_xRefract = static_cast<float *>(mxGPUGetData(xRefract));
    dev_nIter    = static_cast<int *>(mxGPUGetData(nIter));
    dev_zElem    = static_cast<float const *>(mxGPUGetDataReadOnly(zElem));
    dev_xElem    = static_cast<float const *>(mxGPUGetDataReadOnly(xElem));
    dev_zPix     = static_cast<float const *>(mxGPUGetDataReadOnly(zPix));
    dev_xPix     = static_cast<float const *>(mxGPUGetDataReadOnly(xPix));
    
    /* set constant memory */
    if(nElem > 256) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "xElem is too long, kernel supports xElem of up to 256 elements");
    }
    cudaMemcpyToSymbol(   zElemConst, dev_zElem,    nElem*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(   xElemConst, dev_xElem,    nElem*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    /* Execute CUDA kernel */
    refr<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(     dev_xRefract, dev_nIter, 
                                                                  dev_zPix, dev_xPix, 
                                                                  sosInterf, sosSample, timePrec, 
                                                                  nZPix, nXPix, nElem);
    
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(xRefract);
    plhs[1] = mxGPUCreateMxArrayOnGPU(nIter);
    
    /* Clean-up */
    mxGPUDestroyGPUArray(xRefract);
    mxGPUDestroyGPUArray(nIter);
    mxGPUDestroyGPUArray(zElem);
    mxGPUDestroyGPUArray(xElem);
    mxGPUDestroyGPUArray(zPix);
    mxGPUDestroyGPUArray(xPix);
    
    //cudaDeviceReset();
}
