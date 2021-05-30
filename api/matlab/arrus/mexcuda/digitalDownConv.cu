#define M_PI 3.14159265358979
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <iostream>
#include "utils.h"

__constant__ float filtCoeffConst[1024];

__global__ void digitalDownConv(float2 * iqOut, float const * rfIn, 
                                float const fs, float const fn, int const dec, 
                                int const nSampIn, int const nSampOut, int const nCoeff)
{
    int iSampStart = threadIdx.x;
    int iSampStep = blockDim.x;
    int iLine = blockIdx.y;
    
    float modSin, modCos;
    float2 iqAux;
    int iSampIn;
    
    float const deltaPhase = - 2 * M_PI * fn / fs;
    
    // Demodulation
    extern __shared__ float2 iqLineShared[];
    for (int iSamp=iSampStart; iSamp<nSampIn; iSamp+=iSampStep) {
        __sincosf(deltaPhase * iSamp, &modSin, &modCos);
        
        iqLineShared[iSamp].x = 2 * rfIn[iSamp + iLine*nSampIn] * modCos;
        iqLineShared[iSamp].y = 2 * rfIn[iSamp + iLine*nSampIn] * modSin;
    }
    __syncthreads();
    
    // Filtration and decimation
    for (int iSampOut=iSampStart; iSampOut<nSampOut; iSampOut+=iSampStep) {
        iqAux.x = 0.f;
        iqAux.y = 0.f;
        for (int iCoeff=0; iCoeff<nCoeff; iCoeff++) {
            iSampIn = iSampOut*dec + iCoeff;
            iqAux.x += iqLineShared[iSampIn].x * filtCoeffConst[iCoeff];
            iqAux.y += iqLineShared[iSampIn].y * filtCoeffConst[iCoeff];
        }
        iqOut[iSampOut + iLine*nSampOut] = iqAux;
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
    mxGPUArray * iqOut;
    mxGPUArray const * rfIn;
    mxGPUArray const * filtCoeff;
    
    float2 * dev_iqOut;
    float const * dev_rfIn;
    float const * dev_filtCoeff;
    
    float fs;
    float fn;
    int dec;
    
    int nSampIn;
    int nSampOut;
    int nElem;
    int nRep;
    int nCoeff;
    
    dim3 const threadsPerBlock = {256, 1, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
    char const * const invalidInputMsgId = "digitalDownConv:InvalidInput";
    char const * const invalidOutputMsgId = "digitalDownConv:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=5) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "5 inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }
    
    for (int i=2; i<5; i++) {
        if (!mxIsSingle(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
            mexErrMsgIdAndTxt( invalidInputMsgId, "Last 3 inputs must be single, real scalars");
        }
    }
    
    /* Extract inputs from prhs */
    rfIn      = mxGPUCreateFromMxArray(prhs[0]);
    filtCoeff = mxGPUCreateFromMxArray(prhs[1]);
    
    fs    = mxGetScalar(prhs[2]);
    fn    = mxGetScalar(prhs[3]);
    dec   = static_cast<int>(mxGetScalar(prhs[4]));
    
    /* Validate inputs */
    checkData(rfIn,      "rfIn",      false, 3, invalidInputMsgId);
    checkData(filtCoeff, "filtCoeff", false, 1, invalidInputMsgId);
    
    /* Get some additional information */
    nCoeff = mxGPUGetNumberOfElements(filtCoeff);
    nSampIn = mxGPUGetDimensions(rfIn)[0];
    nElem = mxGPUGetDimensions(rfIn)[1];
    if (mxGPUGetNumberOfDimensions(rfIn)<3) {
        nRep = 1;
    }
    else {
        nRep = mxGPUGetDimensions(rfIn)[2];
    }
    nSampOut = 1 + (nSampIn - nCoeff) / dec;
    
    sharedPerBlock = nSampIn*sizeof(float2);
    blocksPerGrid = {1, static_cast<unsigned int>(nElem*nRep), 1};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 3;
    mwSize dimOut[3] = {nSampOut, nElem, nRep};
    
    iqOut = mxGPUCreateGPUArray(nDimOut,
                                dimOut,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_DO_NOT_INITIALIZE);
    
    /* Get pointers on the device */
    dev_iqOut = static_cast<float2 *>(mxGPUGetData(iqOut));
    dev_rfIn  = static_cast<float const *>(mxGPUGetDataReadOnly(rfIn));
    dev_filtCoeff = static_cast<float const *>(mxGPUGetDataReadOnly(filtCoeff));
    
    /* set constant memory */
    if(nCoeff > 1024) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "filtCoeff is too long, kernel supports filtCoeff of up to 1024 elements");
    }
    CUDA_ASSERT(cudaMemcpyToSymbol(filtCoeffConst, dev_filtCoeff, nCoeff*sizeof(float), 0, cudaMemcpyDeviceToDevice));
    
    /* Execute CUDA kernel */
    digitalDownConv<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqOut, dev_rfIn, 
                                                                        fs, fn, dec, 
                                                                        nSampIn, nSampOut, nCoeff);
    CUDA_ASSERT(cudaGetLastError());
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(iqOut);
    
    /* Destroy the mxGPUArray objects */
    mxGPUDestroyGPUArray(iqOut);
    mxGPUDestroyGPUArray(rfIn);
    mxGPUDestroyGPUArray(filtCoeff);
    
    //cudaDeviceReset();
}
