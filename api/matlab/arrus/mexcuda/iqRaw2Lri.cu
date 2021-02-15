#define M_PI 3.14159265358979
// #define _USE_MATH_DEFINES
// #include "math.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"

// check size of the shared memory
// check iSamp+1>=nSamp?

// convex

__constant__ float xElemConst[1024];

// texture <float2, cudaTextureType1D, cudaReadModeElementType> sinCosTex;
// texture <float, cudaTextureType1D, cudaReadModeElementType> rxApodTex;

__forceinline__ __device__ float ownHypotf(float x, float y)
{
    return sqrtf(x*x + y*y);
}


__global__ void iqRaw2Lri(  float2 * iqLri, float2 const * iqRaw, 
                            float const * zPix, float const * xPix, 
                            float const sos, float const fs, float const fn, 
                            float const txFoc, float const txAng, 
                            float const txApCent, float const initDel, 
                            float const minRxTang, float const maxRxTang, 
                            int const nSamp, int const nElem, 
                            int const nZPix, int const nXPix)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    float txDist, rxDist, txTang, rxTang, txApod, rxApod, time, iSamp, iSampMod, interpWgh;
    float modSin, modCos, sampRe, sampIm, pixRe = 0.f, pixIm = 0.f, pixWgh = 0.f;
    float const sosInv = 1 / sos;
    float zDistInv, rngRxTangInv;
    int offset;
    
    if (z>=nZPix || x>=nXPix) {
        return;
    }
    
    
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
    
    zDistInv = 1 / zPix[z];
    rngRxTangInv = 1 / (maxRxTang - minRxTang);
    
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

        
        rxDist = ownHypotf(zPix[z], xPix[x] - xElemConst[iElem]);   // 40us
        rxTang = (xPix[x] - xElemConst[iElem]) * zDistInv;          // 4us
        rxApod = (rxTang >= minRxTang && rxTang <= maxRxTang) ? 1.f : 0.f;
//         rxApod = tex1D(rxApodTex, (rxTang-minRxTang)*rngRxTangInv);
        
        time = (txDist + rxDist) * sosInv + initDel;
        iSamp = time * fs;
        
        
        if (iSamp>=0.f && iSamp<=(float)nSamp) {
            offset = iElem * nSamp;
            interpWgh = modff(iSamp, &iSamp);
            
            iSampMod = time * fn;
            iSampMod = (iSampMod - truncf(iSampMod))*(float)nSin;
            modSin = sincosShared[(int)iSampMod].x;      // 85us
            modCos = sincosShared[(int)iSampMod].y;
            
//             float2 sincos = tex1D(sinCosTex, iSampMod);
//             modSin = sincos.x;
//             modCos = sincos.y;
            
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
    
    mxGPUArray const * rxApod;
    mxGPUArray const * sinCos;
    
    float2 * dev_iqLri;
    float2 const * dev_iqRaw;
    float const * dev_xElem;
    float const * dev_zPix;
    float const * dev_xPix;
    
    float const * dev_rxApod;
    float2 const * dev_sinCos;
    
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
    
    dim3 const threadsPerBlock = {32, 32, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
    char const * const invalidInputMsgId = "iqRaw2Lri:InvalidInput";
    char const * const invalidOutputMsgId = "iqRaw2Lri:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=15) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "15 inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }
    
    for (int i=0; i<6; i++) {
        if (!(mxIsGPUArray(prhs[i]))) {
            mexErrMsgIdAndTxt( invalidInputMsgId, "First 6 inputs must be gpuArray");
        }
    }
    
    for (int i=6; i<15; i++) {
        if (!mxIsSingle(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
            mexErrMsgIdAndTxt( invalidInputMsgId, "Last 9 inputs must be single, real scalars");
        }
    }
    
    
    /* Extract inputs from prhs */
    iqRaw = mxGPUCreateFromMxArray(prhs[0]);
    xElem = mxGPUCreateFromMxArray(prhs[1]);
    zPix  = mxGPUCreateFromMxArray(prhs[2]);
    xPix  = mxGPUCreateFromMxArray(prhs[3]);
    
    rxApod = mxGPUCreateFromMxArray(prhs[4]);
    sinCos = mxGPUCreateFromMxArray(prhs[5]);
    
    sos   = mxGetScalar(prhs[6]);
    fs    = mxGetScalar(prhs[7]);
    fn    = mxGetScalar(prhs[8]);
    
    foc   = mxGetScalar(prhs[9]);
    ang   = mxGetScalar(prhs[10]);
    cent  = mxGetScalar(prhs[11]);
    initDel	= mxGetScalar(prhs[12]);
    minRxTang = mxGetScalar(prhs[13]);
    maxRxTang = mxGetScalar(prhs[14]);
    
    
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
    
    if ((mxGPUGetClassID(rxApod) != mxSINGLE_CLASS) || mxGPUGetComplexity(rxApod) || !(mxGPUGetNumberOfDimensions(rxApod) == 2 && mxGPUGetDimensions(rxApod)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "rxApod must be single, real, horizontal vector.");
    }
    
    if ((mxGPUGetClassID(sinCos) != mxSINGLE_CLASS) || !mxGPUGetComplexity(sinCos) || !(mxGPUGetNumberOfDimensions(sinCos) == 2 && mxGPUGetDimensions(sinCos)[0] == 1)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "sinCos must be single, complex, horizontal vector.");
    }
    
    if (mxGPUGetDimensions(iqRaw)[1] != mxGPUGetNumberOfElements(xElem)) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "size(iqRaw,2) must be equal to length(xElem).");
    }
    
    /* Get some additional information */
    nSamp = mxGPUGetDimensions(iqRaw)[0];
    nElem = mxGPUGetNumberOfElements(xElem);
    nZPix = mxGPUGetNumberOfElements(zPix);
    nXPix = mxGPUGetNumberOfElements(xPix);
    
    int nRxApod = mxGPUGetNumberOfElements(rxApod);
    int nSinCos = mxGPUGetNumberOfElements(sinCos);
    
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
    
    dev_rxApod = (float const *)(mxGPUGetDataReadOnly(rxApod));
    dev_sinCos = (float2 const *)(mxGPUGetDataReadOnly(sinCos));
    
    /* set constant memory */
    if(nElem > 1024) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "xElem is too long, kernel supports xElem of up to 1024 elements");
    }
    cudaMemcpyToSymbol(xElemConst, dev_xElem, nElem*4, 0, cudaMemcpyDeviceToDevice);
    
    /* configure shared memory */
//     cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
    /* configure texture reference */
    size_t * offset;
    
    
//     rxApodTex.normalized  = true;
//     rxApodTex.addressMode[0] = cudaAddressModeBorder;
//     rxApodTex.filterMode  = cudaFilterModeLinear;
//     
//     cudaChannelFormatDesc channelDesc0 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//     cudaBindTexture(offset, rxApodTex, dev_rxApod, channelDesc0, nRxApod*4);
//     if (*offset != 0) {
//         mexErrMsgIdAndTxt( "texture", "offset");
//     }
//     
//     
//     sinCosTex.normalized  = true;
//     sinCosTex.addressMode[0] = cudaAddressModeWrap;
//     sinCosTex.filterMode  = cudaFilterModeLinear;
//     
//     cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//     cudaBindTexture(offset, sinCosTex, dev_sinCos, channelDesc1, nSinCos*8);
//     if (*offset != 0) {
//         mexErrMsgIdAndTxt( "texture", "offset");
//     }
    
    /* Execute CUDA kernel */
    iqRaw2Lri<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqLri, dev_iqRaw, dev_zPix, dev_xPix, 
                                                                  sos, fs, fn, foc, ang, cent, initDel, minRxTang, maxRxTang, 
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
