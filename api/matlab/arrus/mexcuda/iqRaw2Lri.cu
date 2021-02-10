#define M_PI 3.14159265358979
// #define _USE_MATH_DEFINES
// #include "math.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"

// check size of the shared memory
// check iSamp+1>=nSamp?

// convex

__constant__ float xElemConst[1024];


__device__ float ownHypotf(float x, float y)
{
    return sqrtf(x*x + y*y);
}

__global__ void iqRaw2Lri(  float2 * iqLri, float2 const * iqRaw, 
                            float const * zPix, float const * xPix, 
                            float const sos, float const fs, float const fn, 
                            float const txFoc, float const txAng, 
                            float const txApCent, float const maxTang, 
                            int const nSamp, int const nElem, 
                            int const nZPix, int const nXPix)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    float txDist, rxDist, txTang, rxTang, txApod, rxApod, time, iSamp, iSampMod, interpWgh;
    float modSin, modCos, sampRe, sampIm, pixRe = 0.f, pixIm = 0.f, pixWgh = 0.f;
    float const sosInv = 1 / sos;
    float zDistInv;
    int offset;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    
    if (!isinf(txFoc)) {
        /* STA */
        float xFoc	= txFoc * sinf(txAng) + txApCent;
        float zFoc	= txFoc * cosf(txAng);
        
        txDist	= ownHypotf(zPix[z] - zFoc, xPix[x] - xFoc);
        //txDist	= txDist * sign(zPix[z] - zFoc) + txFoc;          // WARNING: sign()=0 => invalid txDist value
        txTang	= abs((xPix[x] - xFoc)) / max(abs(zPix[z] - zFoc),1e-12);
        txApod	= (fabsf(txTang) <= maxTang) ? 1.f : 0.f;
    }
    else {
        /* PWI */
        float r1 = (xPix[x]-xElemConst[0      ]) * cosf(txAng) - zPix[z] * sinf(txAng);
        float r2 = (xPix[x]-xElemConst[nElem-1]) * cosf(txAng) - zPix[z] * sinf(txAng);
        
        txDist = xPix[x] * sinf(txAng) + zPix[z] * cosf(txAng);
        txApod = (r1 >= 0.f && r2 <= 0.f) ? 1.f : 0.f;
    }
    
    zDistInv = 1 / zPix[z];
    
    const int nSin = 512;
    float dPhase = 2 * M_PI / (nSin-1);
    float currentPhase;
    __shared__ float2 sincosShared[nSin];
    int startSamp = threadIdx.x + threadIdx.y * blockDim.x;
    int stepSamp = blockDim.x * blockDim.y;
    
    for (int iSin=startSamp; iSin<nSin; iSin+=stepSamp) {
        currentPhase = iSin * dPhase;
        sincosShared[iSin].x = sinf(currentPhase);
        sincosShared[iSin].y = cosf(currentPhase);
    }
	__syncthreads();
    
    
//     extern __shared__ float2 iqRawShared[];
//     int startSamp = threadIdx.x + threadIdx.y * blockDim.x;
//     int stepSamp = blockDim.x * blockDim.y;
    
    for (int iElem=0; iElem<nElem; iElem++) {
        
//         for (int jSamp=startSamp; jSamp<nSamp; jSamp+=stepSamp) {
//             iqRawShared[jSamp] = iqRaw[jSamp + iElem*nSamp];
//         }
//         __syncthreads();

        
        rxDist = ownHypotf(zPix[z], xPix[x] - xElemConst[iElem]);
        rxTang = (xPix[x] - xElemConst[iElem]) * zDistInv;
        rxApod = (fabsf(rxTang) <= 0.5f) ? 1.f : 0.f;
        // 130us
        
        time = (txDist + rxDist) * sosInv;                         // 90us
        iSamp = time * fs;
        
        
        if (iSamp>=0.f && iSamp<=(float)nSamp) {
            offset = iElem * nSamp;
            interpWgh = modff(iSamp, &iSamp);
            
            iSampMod = time * fn;
            iSampMod = (iSampMod - truncf(iSampMod))*(float)nSin;
            modSin = sincosShared[(int)iSampMod].x;      // 120us
            modCos = sincosShared[(int)iSampMod].y;
            
            sampRe = (iqRaw[offset + (int)iSamp  ].x * (1 - interpWgh)  // 60us
                    + iqRaw[offset + (int)iSamp+1].x *      interpWgh );
            sampIm = (iqRaw[offset + (int)iSamp  ].y * (1 - interpWgh)
                    + iqRaw[offset + (int)iSamp+1].y *      interpWgh );
            
//             sampRe = (iqRawShared[(int)iSamp  ].x * (1 - interpWgh)
//                     + iqRawShared[(int)iSamp+1].x *      interpWgh );
//             sampIm = (iqRawShared[(int)iSamp  ].y * (1 - interpWgh)
//                     + iqRawShared[(int)iSamp+1].y *      interpWgh );
            
            pixRe += (sampRe * modCos - sampIm * modSin) * rxApod; // 60-80us
            pixIm += (sampRe * modSin + sampIm * modCos) * rxApod;
            pixWgh += rxApod;
        }
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
    float2 const * dev_iqRaw;
    float const * dev_xElem;
    float const * dev_zPix;
    float const * dev_xPix;
    
    float sos;
    float fs;
    float fn;
    
    float foc;
    float ang;
    float cent;
    float maxTang;
    
    int nSamp;
    int nElem;
    int nZPix;
    int nXPix;
    
    dim3 const threadsPerBlock = {32, 32, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
    char const * const invalidInputMsgId = "iqRaw2Lri:InvalidInput";
    char const * const invalidOutputMsgId = "iqRaw2Lri:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=11) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "Seven inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }
    
    if (!(mxIsGPUArray(prhs[0])) || 
        !(mxIsGPUArray(prhs[1])) || 
        !(mxIsGPUArray(prhs[2])) || 
        !(mxIsGPUArray(prhs[3]))) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "First 4 inputs (iqRaw, xElem, zPix, xPix) must be gpuArray");
    }
    
    if (!mxIsSingle(prhs[ 4]) || mxIsComplex(prhs[ 4]) || mxGetNumberOfElements(prhs[ 4]) != 1 || 
        !mxIsSingle(prhs[ 5]) || mxIsComplex(prhs[ 5]) || mxGetNumberOfElements(prhs[ 5]) != 1 || 
        !mxIsSingle(prhs[ 6]) || mxIsComplex(prhs[ 6]) || mxGetNumberOfElements(prhs[ 6]) != 1 || 
        !mxIsSingle(prhs[ 7]) || mxIsComplex(prhs[ 7]) || mxGetNumberOfElements(prhs[ 7]) != 1 || 
        !mxIsSingle(prhs[ 8]) || mxIsComplex(prhs[ 8]) || mxGetNumberOfElements(prhs[ 8]) != 1 || 
        !mxIsSingle(prhs[ 9]) || mxIsComplex(prhs[ 9]) || mxGetNumberOfElements(prhs[ 9]) != 1 || 
        !mxIsSingle(prhs[10]) || mxIsComplex(prhs[10]) || mxGetNumberOfElements(prhs[10]) != 1) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "Last 3 inputs (sos, fs, fn) must be single, real scalars");
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
    maxTang	= mxGetScalar(prhs[10]);
    
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
//     sharedPerBlock = (int)(nSamp*8);
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
    
    /* set constant memory */
    if(nElem > 1024) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "xElem is too long, kernel supports xElem of up to 1024 elements");
    }
    cudaMemcpyToSymbol(xElemConst, dev_xElem, nElem*4, 0, cudaMemcpyDeviceToDevice);
    
    /* Execute CUDA kernel */
//     cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
    iqRaw2Lri<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqLri, dev_iqRaw, dev_zPix, dev_xPix, 
                                                                  sos, fs, fn, foc, ang, cent, maxTang, 
                                                                  nSamp, nElem, nZPix, nXPix);
    
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
