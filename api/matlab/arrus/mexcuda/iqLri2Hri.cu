#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <iostream>

__global__ void iqLri2Hri(  float2 * iqHri, 
                            float2 const * iqLri, 
                            float const * iqLriWgh, 
                            int const nPix, 
                            int const nTx)
{
    int iPix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (iPix>=nPix) {
        return;
    }
    
    float pixRe = 0.f;
    float pixIm = 0.f;
    float pixWgh = 0.f;
    
    for (int iTx=0; iTx<nTx; iTx++) {
        float2 iq = iqLri[iPix + iTx*nPix];
        float wgh = iqLriWgh[iPix + iTx*nPix];

        if (isnan(iq.x) || isnan(iq.y) || isnan(wgh)) {
            continue;
        }
        
        pixRe += iq.x * wgh;
        pixIm += iq.y * wgh;
        pixWgh += wgh;
    }
    iqHri[iPix].x = pixRe / pixWgh;
    iqHri[iPix].y = pixIm / pixWgh;
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
    mxGPUArray * iqHri;
    mxGPUArray const * iqLri;
    mxGPUArray const * iqLriWgh;
    
    float2 * dev_iqHri;
    float2 const * dev_iqLri;
    float const * dev_iqLriWgh;

    int nZPix;
    int nXPix;
    int nPix;
    int nTx;
    
    dim3 const threadsPerBlock = {256, 1, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
    char const * const invalidInputMsgId = "iqLri2Hri:InvalidInput";
    char const * const invalidOutputMsgId = "iqLri2Hri:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=2) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "2 inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }
    
    /* Extract inputs from prhs */
    iqLri     = mxGPUCreateFromMxArray(prhs[0]);
    iqLriWgh  = mxGPUCreateFromMxArray(prhs[1]);
    
    /* Validate inputs */
    checkData(iqLri,     "iqLri",     false, true,  3, invalidInputMsgId);
    checkData(iqLriWgh,  "iqLriWgh",  false, false, 3, invalidInputMsgId);
    
    /* Get some additional information */
    nZPix = mxGPUGetDimensions(iqLri)[0];
    nXPix = mxGPUGetDimensions(iqLri)[1];
    nPix = nZPix * nXPix;

    if (mxGPUGetNumberOfDimensions(iqLri)<3) {
        nTx = 1;
    }
    else {
        nTx   = mxGPUGetDimensions(iqLri)[2];
    }

    if (mxGPUGetDimensions(iqLriWgh)[0] != nZPix ||
        mxGPUGetDimensions(iqLriWgh)[1] != nXPix ||
        mxGPUGetDimensions(iqLriWgh)[2] != nTx) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "iqLri and iqLriWgh must be the same size");
    }
    
    sharedPerBlock = 0;
    blocksPerGrid = {(nPix+threadsPerBlock.x-1)/threadsPerBlock.x, 1, 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 2;
    mwSize dimOut[2] = {nZPix, nXPix};
    
    iqHri = mxGPUCreateGPUArray(nDimOut,
                                dimOut,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_iqHri    = static_cast<float2 *>(mxGPUGetData(iqHri));
    dev_iqLri    = static_cast<float2 const *>(mxGPUGetDataReadOnly(iqLri));
    dev_iqLriWgh = static_cast<float const *>(mxGPUGetDataReadOnly(iqLriWgh));
    
    /* Execute CUDA kernel */
    iqLri2Hri<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqHri, 
                                                                  dev_iqLri, 
                                                                  dev_iqLriWgh, 
                                                                  nPix, nTx);
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(iqHri);
    
    /* Clean-up */
    mxGPUDestroyGPUArray(iqHri);
    mxGPUDestroyGPUArray(iqLri);
    mxGPUDestroyGPUArray(iqLriWgh);
    
    //cudaDeviceReset();
}
