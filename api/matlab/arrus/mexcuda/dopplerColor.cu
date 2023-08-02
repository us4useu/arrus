#include "mex.h"
#include "gpu/mxGPUArray.h"

__global__ void dopplerColor(float * color, 
                             float * power, 
                             float * turbu, 
                             float2 const * iqImg, 
                             int const nZPix, 
                             int const nXPix, 
                             int const nRep)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    float2 iqPixCurr, iqPixPrev;
    float auxPower;
    float2 auxCorr = {0.f, 0.f};
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    /* Color & Power estimation */
    iqPixCurr = iqImg[z + x * nZPix];
    auxPower = iqPixCurr.x * iqPixCurr.x + iqPixCurr.y * iqPixCurr.y;
    for (int iRep=1; iRep<nRep; iRep++) {
        iqPixPrev = iqPixCurr;
        iqPixCurr = iqImg[z + x * nZPix + iRep * nZPix * nXPix];
        
        auxPower += iqPixCurr.x * iqPixCurr.x + iqPixCurr.y * iqPixCurr.y;
        auxCorr.x += iqPixCurr.x * iqPixPrev.x + iqPixCurr.y * iqPixPrev.y;
        auxCorr.y += iqPixCurr.y * iqPixPrev.x - iqPixCurr.x * iqPixPrev.y;
    }
    color[z + x*nZPix] = atan2f(auxCorr.y, auxCorr.x);
    power[z + x*nZPix] = auxPower / static_cast<float>(nRep);
    turbu[z + x*nZPix] = 1.f - hypotf(auxCorr.x, auxCorr.y) / auxPower 
                       * static_cast<float>(nRep) / (static_cast<float>(nRep) - 1);
}



void mexFunction(int nlhs, mxArray * plhs[],
                 int nrhs, mxArray const * prhs[])
{
    /* Initialize the GPU API. */
    mxInitGPU();
    
    /* Declare the variables */
    mxGPUArray * color;
    mxGPUArray * power;
    mxGPUArray * turbu;
    mxGPUArray const * iqImg;
    
    float * dev_color;
    float * dev_power;
    float * dev_turbu;
    float2 const * dev_iqImg;
    
    int nZPix;
    int nXPix;
    int nRep;
    
    dim3 const threadsPerBlock = {32, 32, 1};
    dim3 blocksPerGrid;
    
    char const * const invalidInputMsgId = "dopplerColor:InvalidInput";
    char const * const invalidOutputMsgId = "dopplerColor:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=1) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "One input required");
    }
    
    if (nlhs>3) {
        mexErrMsgIdAndTxt(invalidOutputMsgId, "Three outputs allowed");
    }
    
    if (!(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "Input must be gpuArray object containing single, complex 3D array");
    }
    
    /* Extract inputs from prhs */
    iqImg = mxGPUCreateFromMxArray(prhs[0]);
    
    /* Validate inputs */
    if( mxGPUGetClassID(iqImg) != mxSINGLE_CLASS || 
       !mxGPUGetComplexity(iqImg) || 
        mxGPUGetNumberOfDimensions(iqImg) != 3) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "Input must be gpuArray object containing single, complex 3D array");
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
    
    color = mxGPUCreateGPUArray(nDimOut, dimOut, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    power = mxGPUCreateGPUArray(nDimOut, dimOut, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    turbu = mxGPUCreateGPUArray(nDimOut, dimOut, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_color = (float *)(mxGPUGetData(color));
    dev_power = (float *)(mxGPUGetData(power));
    dev_turbu = (float *)(mxGPUGetData(turbu));
    dev_iqImg = (float2 const *)(mxGPUGetDataReadOnly(iqImg));
    
    /* Execute CUDA kernel */
    dopplerColor<<<blocksPerGrid, threadsPerBlock>>>(dev_color, dev_power, dev_turbu, dev_iqImg, nZPix, nXPix, nRep);
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(color);
    plhs[1] = mxGPUCreateMxArrayOnGPU(power);
    plhs[2] = mxGPUCreateMxArrayOnGPU(turbu);
    
    /* Destroy the mxGPUArray objects */
    mxGPUDestroyGPUArray(color);
    mxGPUDestroyGPUArray(power);
    mxGPUDestroyGPUArray(turbu);
    mxGPUDestroyGPUArray(iqImg);
    
}
