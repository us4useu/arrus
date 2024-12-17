#define OEM_N_CHAN 32
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <iostream>

/*
 * Data reorganization and short->float conversion.
 *
 * NOTE:
 * - the block of threads must have OEM_N_CHAN x OEM_N_CHAN size (must be square for transposition)
 * - is operation on float2 space coalescent?
 */

__global__ void rawReorgCplx( float2 *out, 
                              const short *in, 
                              const int *reorgMap, 
                              const unsigned nSampOut, 
                              const int blockOffset)
{
    __shared__ float2 tileShrd[OEM_N_CHAN][OEM_N_CHAN];
    
    size_t idx;
    
    // Input indices
    int iChanIn = threadIdx.x; // gridSize[0] = 1 -> blockIdx.x = 0
    int iSampIn = (blockIdx.y + blockOffset) * blockDim.y + threadIdx.y; // real and imag pair as a single sample
    
    // Copy input to shared
    idx = iSampIn*2*OEM_N_CHAN + iChanIn;
    
    tileShrd[threadIdx.y][threadIdx.x].x = static_cast<float>(in[idx]);
    tileShrd[threadIdx.y][threadIdx.x].y = static_cast<float>(in[idx + OEM_N_CHAN]);
    
    __syncthreads();
    
    // Output indices
    int iSampAux = (blockIdx.y + blockOffset) * blockDim.y + threadIdx.x;
    int iChanAux = threadIdx.y;
    
    int iChunkIn = iSampAux / nSampOut;
    int iSampOut = iSampAux % nSampOut;
    int iChanOut = reorgMap[iChunkIn*OEM_N_CHAN + iChanAux]; // is this read from reorgMap beneficial?
    
    if (iChanOut < 0) {
        return;
    }
    
    // Copy shared to output
    idx = iChanOut*nSampOut + iSampOut;
    
    out[idx].x = tileShrd[threadIdx.x][threadIdx.y].x;
    out[idx].y = tileShrd[threadIdx.x][threadIdx.y].y;
}



__global__ void rawReorgReal( float *out, 
                              const short *in, 
                              const int *reorgMap, 
                              const unsigned nSampOut, 
                              const int blockOffset)
{
    __shared__ float tileShrd[OEM_N_CHAN][OEM_N_CHAN];
    
    size_t idx;
    
    // Input indices
    int iChanIn = threadIdx.x;
    int iSampIn = (blockIdx.y + blockOffset) * blockDim.y + threadIdx.y;
    
    // Copy input to shared
    idx = iSampIn*OEM_N_CHAN + iChanIn;
    
    tileShrd[threadIdx.y][threadIdx.x] = static_cast<float>(in[idx]);
    
    __syncthreads();
    
    // Output indices
    int iSampAux = (blockIdx.y + blockOffset) * blockDim.y + threadIdx.x;
    int iChanAux = threadIdx.y;
    
    int iChunkIn = iSampAux / nSampOut;
    int iSampOut = iSampAux % nSampOut;
    int iChanOut = reorgMap[iChunkIn*OEM_N_CHAN + iChanAux]; // is this read from reorgMap beneficial?
    
    if (iChanOut < 0) {
        return;
    }
    
    // Copy shared to output
    idx = iChanOut*nSampOut + iSampOut;
    
    out[idx] = tileShrd[threadIdx.x][threadIdx.y];
}



void mexFunction(int nlhs, mxArray * plhs[],
                 int nrhs, mxArray const * prhs[])
{
    /* Initialize the GPU API. */
    mxInitGPU();
    
    /* Declare the variables */
    mxGPUArray * out;
    mxGPUArray const * in;
    mxGPUArray const * reorgMap;
    
    short const * dev_in;
    int const * dev_reorgMap;
    
    unsigned int nRx;
    unsigned int nTx;
    unsigned int nRep;
    bool isComplex;
    
    char const * const invalidInputMsgId = "rawReorg:InvalidInput";
    char const * const invalidOutputMsgId = "rawReorg:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=6) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "6 inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }

    /* Extract inputs from prhs */
    in         = mxGPUCreateFromMxArray(prhs[0]);
    reorgMap   = mxGPUCreateFromMxArray(prhs[1]);
    nRx        = mxGetScalar(prhs[2]);
    nTx        = mxGetScalar(prhs[3]);
    nRep       = mxGetScalar(prhs[4]);
    isComplex  = mxGetScalar(prhs[5]);
    
    /* Validate inputs */
    
    
    /* Get some additional information */
    unsigned int nSampIn = mxGPUGetDimensions(in)[1];
    unsigned int nChunkIn = mxGPUGetDimensions(reorgMap)[1];
    
    /* Get input pointers on the device */
    dev_in       = static_cast<short const *>(mxGPUGetDataReadOnly(in));
    dev_reorgMap = static_cast<int const *>(mxGPUGetDataReadOnly(reorgMap));
    
    /* Create output mxGPUArray object, get its pointer, and run kernel*/
    dim3 const threadsPerBlock = {OEM_N_CHAN, OEM_N_CHAN, 1};
    
    if (isComplex) {
        unsigned int nSampOut = nSampIn / 2 / nChunkIn;
        
        mwSize nDimOut = 4;
        mwSize dimOut[4] = {nSampOut, nRx, nTx, nRep};
        
        out = mxGPUCreateGPUArray(nDimOut, dimOut, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
        float2 * dev_out = static_cast<float2 *>(mxGPUGetData(out));
        
        /* Kernel in loop - due to blocksPerGrid limit */
        int nBlock = nSampIn / 2 / threadsPerBlock.y;
        int nBlockPerPart = (nBlock <= 65535) ? nBlock : 65535;
        int nPart = (nBlock+nBlockPerPart-1)/nBlockPerPart;
        for (int iPart=0; iPart<nPart; iPart++) {
            
            unsigned int nBlockInThisPart = (iPart<(nPart-1)) ? nBlockPerPart : (nBlock-iPart*nBlockPerPart);
            
            dim3 blocksPerGrid = {1, nBlockInThisPart, 1};
            rawReorgCplx<<<blocksPerGrid, threadsPerBlock>>>( dev_out, dev_in, dev_reorgMap, nSampOut, iPart*nBlockPerPart);
        }
    }
    else {
        unsigned int nSampOut = nSampIn / nChunkIn;
        
        mwSize nDimOut = 4;
        mwSize dimOut[4] = {nSampOut, nRx, nTx, nRep};
        
        out = mxGPUCreateGPUArray(nDimOut, dimOut, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
        float * dev_out = static_cast<float *>(mxGPUGetData(out));
        
        /* Kernel in loop - due to blocksPerGrid limit */
        int nBlock = nSampIn / threadsPerBlock.y;
        int nBlockPerPart = (nBlock <= 65535) ? nBlock : 65535;
        int nPart = (nBlock+nBlockPerPart-1)/nBlockPerPart;
        for (int iPart=0; iPart<nPart; iPart++) {
            
            unsigned int nBlockInThisPart = (iPart<(nPart-1)) ? nBlockPerPart : (nBlock-iPart*nBlockPerPart);

            dim3 blocksPerGrid = {1, nBlockInThisPart, 1};
            rawReorgReal<<<blocksPerGrid, threadsPerBlock>>>( dev_out, dev_in, dev_reorgMap, nSampOut, iPart*nBlockPerPart);
        }
    }
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(out);
    
    /* Clean-up */
    mxGPUDestroyGPUArray(out);
    mxGPUDestroyGPUArray(in);
    mxGPUDestroyGPUArray(reorgMap);
    
    //cudaDeviceReset();
}
