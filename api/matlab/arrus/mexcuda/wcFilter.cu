#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <iostream>

__constant__ float filtNumConst[9];
__constant__ float filtDenConst[9];
__constant__ float filtRectInitConst[8];

__global__ void wcFilter(float2 * output, 
                         float2 * filtState, 
                         float2 const * input, 
                         int const nZPix, 
                         int const nXPix, 
                         int const nRep, 
                         int const filtOrd)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    float2 splIn, splOut;
    float2 flt1, flt2, flt3, flt4, flt5, flt6, flt7, flt8;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    splIn = input[z + x * nZPix];
    
    flt1.x = filtRectInitConst[0] * splIn.x;
    flt1.y = filtRectInitConst[0] * splIn.y;
    flt2.x = filtRectInitConst[1] * splIn.x;
    flt2.y = filtRectInitConst[1] * splIn.y;
    flt3.x = filtRectInitConst[2] * splIn.x;
    flt3.y = filtRectInitConst[2] * splIn.y;
    flt4.x = filtRectInitConst[3] * splIn.x;
    flt4.y = filtRectInitConst[3] * splIn.y;
    flt5.x = filtRectInitConst[4] * splIn.x;
    flt5.y = filtRectInitConst[4] * splIn.y;
    flt6.x = filtRectInitConst[5] * splIn.x;
    flt6.y = filtRectInitConst[5] * splIn.y;
    flt7.x = filtRectInitConst[6] * splIn.x;
    flt7.y = filtRectInitConst[6] * splIn.y;
    flt8.x = filtRectInitConst[7] * splIn.x;
    flt8.y = filtRectInitConst[7] * splIn.y;
    
    for (int iRep=0; iRep<nRep; iRep++) {
        splIn = input[z + x * nZPix + iRep * nZPix * nXPix];

        splOut.x = splIn.x * filtNumConst[0] + flt1.x;
        splOut.y = splIn.y * filtNumConst[0] + flt1.y;
        
        flt1.x = flt2.x + splIn.x * filtNumConst[1] - splOut.x * filtDenConst[1];
        flt1.y = flt2.y + splIn.y * filtNumConst[1] - splOut.y * filtDenConst[1];
        flt2.x = flt3.x + splIn.x * filtNumConst[2] - splOut.x * filtDenConst[2];
        flt2.y = flt3.y + splIn.y * filtNumConst[2] - splOut.y * filtDenConst[2];
        flt3.x = flt4.x + splIn.x * filtNumConst[3] - splOut.x * filtDenConst[3];
        flt3.y = flt4.y + splIn.y * filtNumConst[3] - splOut.y * filtDenConst[3];
        flt4.x = flt5.x + splIn.x * filtNumConst[4] - splOut.x * filtDenConst[4];
        flt4.y = flt5.y + splIn.y * filtNumConst[4] - splOut.y * filtDenConst[4];
        flt5.x = flt6.x + splIn.x * filtNumConst[5] - splOut.x * filtDenConst[5];
        flt5.y = flt6.y + splIn.y * filtNumConst[5] - splOut.y * filtDenConst[5];
        flt6.x = flt7.x + splIn.x * filtNumConst[6] - splOut.x * filtDenConst[6];
        flt6.y = flt7.y + splIn.y * filtNumConst[6] - splOut.y * filtDenConst[6];
        flt7.x = flt8.x + splIn.x * filtNumConst[7] - splOut.x * filtDenConst[7];
        flt7.y = flt8.y + splIn.y * filtNumConst[7] - splOut.y * filtDenConst[7];
        flt8.x =          splIn.x * filtNumConst[8] - splOut.x * filtDenConst[8];
        flt8.y =          splIn.y * filtNumConst[8] - splOut.y * filtDenConst[8];
        
        output[z + x * nZPix + iRep * nZPix * nXPix] = splOut;
    }
    
    filtState[z + x * nZPix + 0 * nZPix * nXPix] = flt1;
    filtState[z + x * nZPix + 1 * nZPix * nXPix] = flt2;
    filtState[z + x * nZPix + 2 * nZPix * nXPix] = flt3;
    filtState[z + x * nZPix + 3 * nZPix * nXPix] = flt4;
    filtState[z + x * nZPix + 4 * nZPix * nXPix] = flt5;
    filtState[z + x * nZPix + 5 * nZPix * nXPix] = flt6;
    filtState[z + x * nZPix + 6 * nZPix * nXPix] = flt7;
    filtState[z + x * nZPix + 7 * nZPix * nXPix] = flt8;
}

__host__ void checkData(mxGPUArray const * const data, 
                        char const * const name, 
                        bool const mustBeComplex, 
                        int const mustBeNDim, 
                        char const * const invalidInputMsgId)
{
    std::string invalidInputMsgTxt(name);
    
    if (mxGPUGetClassID(data) != mxSINGLE_CLASS) 
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
    mxGPUArray * output;
    mxGPUArray * filtState;
    mxGPUArray const * input;
    mxGPUArray const * filtNum;
    mxGPUArray const * filtDen;
    mxGPUArray const * filtRectInit;
    
    float2 * dev_output;
    float2 * dev_filtState;
    float2 const * dev_input;
    float const * dev_filtNum;
    float const * dev_filtDen;
    float const * dev_filtRectInit;
    
    int nZPix;
    int nXPix;
    int nRep;
    int filtOrd;
    
    dim3 const threadsPerBlock = {32, 32, 1};
    dim3 blocksPerGrid;
    
    char const * const invalidInputMsgId = "wcFilter:InvalidInput";
    char const * const invalidOutputMsgId = "wcFilter:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=3 && nrhs!=4) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "Three or four inputs required");
    }
    
    if (nlhs>2) {
        mexErrMsgIdAndTxt(invalidOutputMsgId, "Two outputs allowed");
    }
    
//     if (!(mxIsGPUArray(prhs[0]))) {
//         mexErrMsgIdAndTxt(invalidInputMsgId, "Input must be gpuArray object containing single, complex 3D array");
//     }
    
    /* Extract inputs from prhs */
    input        = mxGPUCreateFromMxArray(prhs[0]);
    filtNum      = mxGPUCreateFromMxArray(prhs[1]);
    filtDen      = mxGPUCreateFromMxArray(prhs[2]);
    filtRectInit = mxGPUCreateFromMxArray(prhs[3]);
    
    /* Validate inputs */
    checkData(input,        "input",        true,   3, invalidInputMsgId);
    checkData(filtNum,      "filtNum",      false,  1, invalidInputMsgId);
    checkData(filtDen,      "filtDen",      false,  1, invalidInputMsgId);
    checkData(filtRectInit, "filtRectInit", false,  1, invalidInputMsgId);
    
    /* Get some additional information */
    nZPix = mxGPUGetDimensions(input)[0];
    nXPix = mxGPUGetDimensions(input)[1];
    nRep = mxGPUGetDimensions(input)[2];
    filtOrd = mxGPUGetNumberOfElements(filtNum) - 1;

    /* Validate filter order */
    if (mxGPUGetNumberOfElements(filtDen) != filtOrd + 1) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "filtNum and filtDen must have the same length");
    }
    if (mxGPUGetNumberOfElements(filtRectInit) != filtOrd) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "filtRectInit length must equal the filter order");
    }
    if (filtOrd<1 || filtOrd>8) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "filter order must be in 1-8 range");
    }
    
    blocksPerGrid = {(nZPix+threadsPerBlock.x-1)/threadsPerBlock.x, 
                     (nXPix+threadsPerBlock.y-1)/threadsPerBlock.y, 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 3;
    mwSize dimOut0[3] = {nZPix, nXPix, nRep};
    mwSize dimOut1[3] = {nZPix, nXPix, filtOrd};
    
    output    = mxGPUCreateGPUArray(nDimOut, dimOut0, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    filtState = mxGPUCreateGPUArray(nDimOut, dimOut1, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_output       = (float2 *)(mxGPUGetData(output));
    dev_filtState    = (float2 *)(mxGPUGetData(filtState));
    dev_input        = (float2 const *)(mxGPUGetDataReadOnly(input));
    dev_filtNum      = (float const *)(mxGPUGetDataReadOnly(filtNum));
    dev_filtDen      = (float const *)(mxGPUGetDataReadOnly(filtDen));
    dev_filtRectInit = (float const *)(mxGPUGetDataReadOnly(filtRectInit));
    
    /* set constant memory */
    if(filtOrd > 8) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "Filter order too high, kernel supports up to 8th filter order");
    }
    cudaMemcpyToSymbol(filtNumConst,      dev_filtNum,      (filtOrd+1)*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(filtDenConst,      dev_filtDen,      (filtOrd+1)*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(filtRectInitConst, dev_filtRectInit,  filtOrd   *sizeof(float), 0, cudaMemcpyDeviceToDevice);

    /* Execute CUDA kernel */
    wcFilter<<<blocksPerGrid, threadsPerBlock>>>(dev_output, 
                                                 dev_filtState, 
                                                 dev_input, 
                                                 nZPix, 
                                                 nXPix, 
                                                 nRep, 
                                                 filtOrd);
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(output);
    plhs[1] = mxGPUCreateMxArrayOnGPU(filtState);
    
    /* Destroy the mxGPUArray objects */
    mxGPUDestroyGPUArray(output);
    mxGPUDestroyGPUArray(filtState);
    mxGPUDestroyGPUArray(input);
    mxGPUDestroyGPUArray(filtNum);
    mxGPUDestroyGPUArray(filtDen);
    mxGPUDestroyGPUArray(filtRectInit);
    
}
