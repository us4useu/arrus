#define _USE_MATH_DEFINES
#include "math.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"

__global__ void dopplerColor(float * color, 
                             float2 const * iqImg, 
                             float const prf, 
                             int const nZPix, 
                             int const nXPix, 
                             int const nRep)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    float2 iqPixCurr, iqPixPrev;
    float auxRe = 0.f;
    float auxIm = 0.f;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    for (int iRep=1; iRep<nRep; iRep++) {
        
        iqPixCurr = iqImg[z + x * nZPix +  iRep    * nZPix * nXPix];
        iqPixPrev = iqImg[z + x * nZPix + (iRep-1) * nZPix * nXPix];
        
        auxRe += iqPixCurr.x * iqPixPrev.x + iqPixCurr.y * iqPixPrev.y;
        auxIm += iqPixCurr.y * iqPixPrev.x - iqPixCurr.x * iqPixPrev.y;
    }
    
    color[z + x*nZPix] = atan2f(auxIm, auxRe) / (2 * M_PI) * prf;
}



void mexFunction(int nlhs, mxArray * plhs[],
                 int nrhs, mxArray const * prhs[])
{
    /* Initialize the GPU API. */
    mxInitGPU();
    
    /* Declare the variables */
    mxGPUArray * color;
    mxGPUArray const * iqImg;
    
    float * dev_color;
    float2 const * dev_iqImg;
    
    float prf;
    int nZPix;
    int nXPix;
    int nRep;
    
    dim3 const threadsPerBlock = {32, 32, 1};
    dim3 blocksPerGrid;
    
    char const * const invalidInputMsgId = "dopplerColor:InvalidInput";
    char const * const invalidOutputMsgId = "dopplerColor:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=2) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "Two inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt(invalidOutputMsgId, "One output allowed");
    }
    
    if (!(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "iqImg must be gpuArray");
    }
    
    if (!mxIsSingle(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "prf must be single, real scalar");
    }
    
    /* Extract inputs from prhs */
    iqImg = mxGPUCreateFromMxArray(prhs[0]);
    
    prf   = mxGetScalar(prhs[1]);
    
    /* Validate inputs */
    if( mxGPUGetClassID(iqImg) != mxSINGLE_CLASS || 
       !mxGPUGetComplexity(iqImg) || 
        mxGPUGetNumberOfDimensions(iqImg) != 3) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "iqImg must be single, complex 3D array.");
    }
    
    /* Get some additional information */
    nZPix = mxGPUGetDimensions(iqImg)[0];
    nXPix = mxGPUGetDimensions(iqImg)[1];
    nRep = mxGPUGetDimensions(iqImg)[2];
    
    blocksPerGrid = {(unsigned int)ceilf((float)nZPix/(float)threadsPerBlock.x), 
                     (unsigned int)ceilf((float)nXPix/(float)threadsPerBlock.y), 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 2;
    mwSize dimOut[2] = {nZPix, nXPix};
    
    color = mxGPUCreateGPUArray(nDimOut,
                                dimOut,
                                mxSINGLE_CLASS,
                                mxREAL,
                                MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_color = (float *)(mxGPUGetData(color));
    dev_iqImg = (float2 const *)(mxGPUGetDataReadOnly(iqImg));
    
    /* Execute CUDA kernel */
    dopplerColor<<<blocksPerGrid, threadsPerBlock>>>(dev_color, dev_iqImg, prf, nZPix, nXPix, nRep);
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(color);
    
    /* Destroy the mxGPUArray objects */
    mxGPUDestroyGPUArray(color);
    mxGPUDestroyGPUArray(iqImg);
    
}
