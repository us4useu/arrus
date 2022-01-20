#define M_PI 3.14159265358979
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <iostream>

__constant__ float xElemConst[256];
__constant__ float yElemConst[256];

texture <float2, cudaTextureType1DLayered, cudaReadModeElementType> iqRawTex;
texture <float, cudaTextureType1D, cudaReadModeElementType> rxApodTex;

__forceinline__ __device__ float ownHypotf(float z, float x)
{
    return sqrtf(z*z + x*x);
}

__forceinline__ __device__ float ownHypotf(float z, float x, float y)
{
    return sqrtf(z*z + x*x + y*y);
}

// if pitchX == pitchY, then min/maxRxTangZX = min/maxRxTangZY
// images are already compounded (coherently)
// the final image is not divided by sum(txApod). This needs to be done in postprocessing.
// maybe dynamic rx apodization should be replaced with static rx apodization?

__global__ void iqRaw2Lri(  float2 * iqLri, float const * zPix, float const * xPix, float const * yPix, 
                            float const * txFoc, float const * txAngZX, float const * txAngZY, 
                            float const * txApCentX, float const * txApCentY, 
                            int const * txApFstElemX, int const * txApLstElemX, 
                            int const * txApFstElemY, int const * txApLstElemY, 
                            int const * nSampOmit, 
                            float const minRxTangZX, float const maxRxTangZX, 
                            float const minRxTangZY, float const maxRxTangZY, 
                            float const fs, float const fn, 
                            float const sos, float const initDel, 
                            int const nZPix, int const nXPix, int const nYPix, 
                            int const nSamp, int const nElemX, int const nElemY, 
                            int const nTx)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z>=nZPix || x>=nXPix || y>=nYPix) {
        return;
    }
    
    float txAngZn, txAngAz;
    float txDist, rxDist, rxTang, txApod, rxApodX, rxApodY, time, iSamp;
    float modSin, modCos, sampRe, sampIm, pixRe, pixIm, pixWgh;
    float const omega = 2 * M_PI * fn;
    float const sosInv = 1 / sos;
    float const zDistInv = 1 / zPix[z];
    float const rngRxTangZXInv = 1 / (maxRxTangZX - minRxTangZX); // inverted zx-tangent range
    float const rngRxTangZYInv = 1 / (maxRxTangZY - minRxTangZY); // inverted zy-tangent range
    
    for (int iTx=0; iTx<nTx; iTx++) {
        
        txAngZn = atanf(ownHypotf(tanf(txAngZX[iTx]), tanf(txAngZY[iTx])));
        txAngAz = atan2f(tanf(txAngZY[iTx]), tanf(txAngZX[iTx]));
        
        if (!isinf(txFoc[iTx])) {
            /* STA */
            float zFoc	= txFoc[iTx] * cosf(txAngZn);
            float xFoc	= txFoc[iTx] * sinf(txAngZn) * cosf(txAngAz) + txApCentX[iTx];
            float yFoc	= txFoc[iTx] * sinf(txAngZn) * sinf(txAngAz) + txApCentY[iTx];
            
            float pixFocArrang;
            
            if (txFoc[iTx] <= 0.f) {
                /* Virtual Point Source BEHIND probe surface */
                // Valid pixels are assumed to be always in front of the focal point (VSP)
                pixFocArrang = 1.f;
            }
            else {
                /* Virtual Point Source IN FRONT OF probe surface */
                // Projection of the Foc-Pix vector on the ApCent-Foc vector (dot product) ...
                // to determine if the pixel is behind (-) or in front of (+) the focal point (VSP).
                pixFocArrang = (((zPix[z]-zFoc)* zFoc + 
                                 (xPix[x]-xFoc)*(xFoc-txApCentX[iTx]) + 
                                 (yPix[x]-yFoc)*(yFoc-txApCentY[iTx])) >= 0.f) ? 1.f : -1.f;
            }
            txDist	= ownHypotf(zPix[z] - zFoc, xPix[x] - xFoc, yPix[y] - yFoc);
            txDist *= pixFocArrang; // Compensation for the Pix-Foc arrangement
            txDist += txFoc[iTx]; // Compensation for the reference time being the moment when txApCent fires.
            
            // Projections of Foc-Pix vector on the rotated Foc-ApEdge vectors (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // Foc-ApEdgeFst vector is rotated left, Foc-ApEdgeLst vector is rotated right.
            txApod = ( ( ( - (zPix[z] - zFoc) * (xElemConst[txApFstElemX[iTx]] - xFoc) 
                           - (xPix[x] - xFoc) * zFoc) * pixFocArrang >= 0.f ) && 
                       ( (   (zPix[z] - zFoc) * (xElemConst[txApLstElemX[iTx]] - xFoc) 
                           + (xPix[x] - xFoc) * zFoc) * pixFocArrang >= 0.f ) && 
                       ( ( - (zPix[z] - zFoc) * (yElemConst[txApFstElemY[iTx]] - yFoc) 
                           - (yPix[x] - yFoc) * zFoc) * pixFocArrang >= 0.f ) && 
                       ( (   (zPix[z] - zFoc) * (yElemConst[txApLstElemY[iTx]] - yFoc) 
                           + (yPix[x] - yFoc) * zFoc) * pixFocArrang >= 0.f ) ) ? 1.f : 0.f;
        }
        else {
            /* PWI */
            txDist =    zPix[z] * cosf(txAngZn) + 
                     ( (xPix[x] - txApCentX[iTx]) * cosf(txAngAz) + 
                       (yPix[y] - txApCentY[iTx]) * sinf(txAngAz) ) * sinf(txAngZn);
            
            // Projections of ApEdge-Pix vector on the rotated unit vector of tx direction (dot products) ...
            // to determine if the pixel is in the sonified area (dot product >= 0).
            // For ApEdgeFst, the vector is rotated left, for ApEdgeLst the vector is rotated right.
            txApod = ( ( ( - sinf(txAngZX[iTx]) *  zPix[z] 
                           + cosf(txAngZX[iTx]) * (xPix[x]-xElemConst[txApFstElemX[iTx]])) >= 0.f ) && 
                       ( (   sinf(txAngZX[iTx]) *  zPix[z] 
                           - cosf(txAngZX[iTx]) * (xPix[x]-xElemConst[txApLstElemX[iTx]])) >= 0.f ) && 
                       ( ( - sinf(txAngZY[iTx]) *  zPix[z] 
                           + cosf(txAngZY[iTx]) * (yPix[y]-yElemConst[txApFstElemY[iTx]])) >= 0.f ) && 
                       ( (   sinf(txAngZY[iTx]) *  zPix[z] 
                           - cosf(txAngZY[iTx]) * (yPix[y]-yElemConst[txApLstElemY[iTx]])) >= 0.f ) ) ? 1.f : 0.f;
        }
        
        pixRe = 0.f;
        pixIm = 0.f;
        pixWgh = 0.f;
        
        if (txApod != 0.f) {
            
            for (int iElemY=0; iElemY<nElemY; iElemY++) {
                
                rxTang = (yPix[y] - yElemConst[iElemY]) * zDistInv;
                if (rxTang < minRxTangZY || rxTang > maxRxTangZY) continue;
                rxApodY = tex1D(rxApodTex, (rxTang-minRxTangZY)*rngRxTangZYInv); // normalized texture fetching, errors at aperture sides
                
                for (int iElemX=0; iElemX<nElemX; iElemX++) {
                    
                    rxTang = (xPix[x] - xElemConst[iElemX]) * zDistInv;
                    if (rxTang < minRxTangZX || rxTang > maxRxTangZX) continue;
                    rxApodX = tex1D(rxApodTex, (rxTang-minRxTangZX)*rngRxTangZXInv); // normalized texture fetching, errors at aperture sides
                    
                    rxDist = ownHypotf(zPix[z], xPix[x] - xElemConst[iElemX], yPix[y] - yElemConst[iElemY]);
                    time = (txDist + rxDist) * sosInv + initDel;
                    iSamp = time * fs;
                    if (iSamp<static_cast<float>(nSampOmit[iTx]) || iSamp>static_cast<float>(nSamp-1)) continue;
                    
                    float2 iqSamp = tex1DLayered(iqRawTex, iSamp + 0.5f, iElemX + iElemY*nElemX + iTx*nElemX*nElemY);
                    sampRe = iqSamp.x;
                    sampIm = iqSamp.y;
                    
                    __sincosf(omega * time, &modSin, &modCos);
                    
                    pixRe += (sampRe * modCos - sampIm * modSin) * rxApodX * rxApodY;
                    pixIm += (sampRe * modSin + sampIm * modCos) * rxApodX * rxApodY;
                    pixWgh += rxApodX * rxApodY;
                }
            }
        }
        
        if (pixWgh != 0.f) {
            iqLri[z + x*nZPix + y*nZPix*nXPix].x += pixRe / pixWgh * txApod;
            iqLri[z + x*nZPix + y*nZPix*nXPix].y += pixIm / pixWgh * txApod;
        }
    }
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
    mxGPUArray * iqLri;
    mxGPUArray const * iqRaw;
    mxGPUArray const * xElem;
    mxGPUArray const * yElem;
    mxGPUArray const * zPix;
    mxGPUArray const * xPix;
    mxGPUArray const * yPix;
    mxGPUArray const * rxApod;
    mxGPUArray const * foc;
    mxGPUArray const * angZX;
    mxGPUArray const * angZY;
    mxGPUArray const * centX;
    mxGPUArray const * centY;
    mxGPUArray const * elemXFst;
    mxGPUArray const * elemXLst;
    mxGPUArray const * elemYFst;
    mxGPUArray const * elemYLst;
    mxGPUArray const * nSampOmit;
    
    float2 * dev_iqLri;
    float2 const * dev_iqRaw;
    float const * dev_xElem;
    float const * dev_yElem;
    float const * dev_zPix;
    float const * dev_xPix;
    float const * dev_yPix;
    float const * dev_rxApod;
    float const * dev_foc;
    float const * dev_angZX;
    float const * dev_angZY;
    float const * dev_centX;
    float const * dev_centY;
    int const * dev_elemXFst;
    int const * dev_elemXLst;
    int const * dev_elemYFst;
    int const * dev_elemYLst;
    int const * dev_nSampOmit;
    
    float minRxTangZX;
    float maxRxTangZX;
    float minRxTangZY;
    float maxRxTangZY;
    float fs;
    float fn;
    float sos;
    float initDel;
    
    int nSamp;
    int nElemX;
    int nElemY;
    int nZPix;
    int nXPix;
    int nYPix;
    int nRx;
    int nTx;
    int nRxApodSamp;
    
    dim3 const threadsPerBlock = {16, 16, 1};
    dim3 blocksPerGrid;
    int sharedPerBlock;
    
    char const * const invalidInputMsgId = "iqRaw2Lri3D:InvalidInput";
    char const * const invalidOutputMsgId = "iqRaw2Lri3D:InvalidOutput";
    
    /* Validate mex inputs/outputs */
    if (nrhs!=25) {
        mexErrMsgIdAndTxt( invalidInputMsgId, "25 inputs required");
    }
    
    if (nlhs>1) {
        mexErrMsgIdAndTxt( invalidOutputMsgId, "One output allowed");
    }
    
//     for (int i=17; i<25; i++) {
//         if (!mxIsSingle(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1) {
//             mexErrMsgIdAndTxt( invalidInputMsgId, "Last 8 inputs must be single, real scalars");
//         }
//     }
    
    
    /* Extract inputs from prhs */
    iqRaw     = mxGPUCreateFromMxArray(prhs[0]);
    xElem     = mxGPUCreateFromMxArray(prhs[1]);
    yElem     = mxGPUCreateFromMxArray(prhs[2]);
    zPix      = mxGPUCreateFromMxArray(prhs[3]);
    xPix      = mxGPUCreateFromMxArray(prhs[4]);
    yPix      = mxGPUCreateFromMxArray(prhs[5]);
    rxApod    = mxGPUCreateFromMxArray(prhs[6]);
    foc       = mxGPUCreateFromMxArray(prhs[7]);
    angZX     = mxGPUCreateFromMxArray(prhs[8]);
    angZY     = mxGPUCreateFromMxArray(prhs[9]);
    centX     = mxGPUCreateFromMxArray(prhs[10]);
    centY     = mxGPUCreateFromMxArray(prhs[11]);
    elemXFst  = mxGPUCreateFromMxArray(prhs[12]);
    elemXLst  = mxGPUCreateFromMxArray(prhs[13]);
    elemYFst  = mxGPUCreateFromMxArray(prhs[14]);
    elemYLst  = mxGPUCreateFromMxArray(prhs[15]);
    nSampOmit = mxGPUCreateFromMxArray(prhs[16]);

    minRxTangZX = mxGetScalar(prhs[17]);
    maxRxTangZX = mxGetScalar(prhs[18]);
    minRxTangZY = mxGetScalar(prhs[19]);
    maxRxTangZY = mxGetScalar(prhs[20]);
    fs        = mxGetScalar(prhs[21]);
    fn        = mxGetScalar(prhs[22]);
    sos       = mxGetScalar(prhs[23]);
    initDel   = mxGetScalar(prhs[24]);
    
    /* Validate inputs */
    checkData(iqRaw,     "iqRaw",     false, true,  3, invalidInputMsgId);
    checkData(xElem,     "xElem",     false, false, 1, invalidInputMsgId);
    checkData(yElem,     "yElem",     false, false, 1, invalidInputMsgId);
    checkData(zPix,      "zPix",      false, false, 1, invalidInputMsgId);
    checkData(xPix,      "xPix",      false, false, 1, invalidInputMsgId);
    checkData(yPix,      "yPix",      false, false, 1, invalidInputMsgId);
    checkData(rxApod,    "rxApod",    false, false, 1, invalidInputMsgId);
    checkData(foc,       "foc",       false, false, 1, invalidInputMsgId);
    checkData(angZX,     "angZX",     false, false, 1, invalidInputMsgId);
    checkData(angZY,     "angZY",     false, false, 1, invalidInputMsgId);
    checkData(centX,     "centX",     false, false, 1, invalidInputMsgId);
    checkData(centY,     "centY",     false, false, 1, invalidInputMsgId);
    checkData(elemXFst,  "elemXFst",  true,  false, 1, invalidInputMsgId);
    checkData(elemXLst,  "elemXLst",  true,  false, 1, invalidInputMsgId);
    checkData(elemYFst,  "elemYFst",  true,  false, 1, invalidInputMsgId);
    checkData(elemYLst,  "elemYLst",  true,  false, 1, invalidInputMsgId);
    checkData(nSampOmit, "nSampOmit", true,  false, 1, invalidInputMsgId);
    
    /* Get some additional information */
    nSamp   = mxGPUGetDimensions(iqRaw)[0];
    nRx     = mxGPUGetDimensions(iqRaw)[1];
    nElemX  = mxGPUGetNumberOfElements(xElem);
    nElemY  = mxGPUGetNumberOfElements(yElem);
    nZPix   = mxGPUGetNumberOfElements(zPix);
    nXPix   = mxGPUGetNumberOfElements(xPix);
    nYPix   = mxGPUGetNumberOfElements(yPix);
    nRxApodSamp = mxGPUGetNumberOfElements(rxApod);
    if (mxGPUGetNumberOfDimensions(iqRaw)<3) {
        nTx = 1;
    }
    else {
        nTx   = mxGPUGetDimensions(iqRaw)[2];
    }
    
    sharedPerBlock = 0;
    blocksPerGrid = {(nZPix+threadsPerBlock.x-1)/threadsPerBlock.x, 
                     (nXPix+threadsPerBlock.y-1)/threadsPerBlock.y, 
                     (nYPix+threadsPerBlock.z-1)/threadsPerBlock.z};
    
    /* Create output mxGPUArray object */
    mwSize nDimOut = 3;
    mwSize dimOut[3] = {nZPix, nXPix, nYPix};
    
    iqLri = mxGPUCreateGPUArray(nDimOut,
                                dimOut,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_INITIALIZE_VALUES);
    
    /* Get pointers on the device */
    dev_iqLri    = static_cast<float2 *>(mxGPUGetData(iqLri));
    dev_iqRaw    = static_cast<float2 const *>(mxGPUGetDataReadOnly(iqRaw));
    dev_xElem    = static_cast<float const *>(mxGPUGetDataReadOnly(xElem));
    dev_yElem    = static_cast<float const *>(mxGPUGetDataReadOnly(yElem));
    dev_zPix     = static_cast<float const *>(mxGPUGetDataReadOnly(zPix));
    dev_xPix     = static_cast<float const *>(mxGPUGetDataReadOnly(xPix));
    dev_yPix     = static_cast<float const *>(mxGPUGetDataReadOnly(yPix));
    dev_rxApod   = static_cast<float const *>(mxGPUGetDataReadOnly(rxApod));
    dev_foc      = static_cast<float const *>(mxGPUGetDataReadOnly(foc));
    dev_angZX    = static_cast<float const *>(mxGPUGetDataReadOnly(angZX));
    dev_angZY    = static_cast<float const *>(mxGPUGetDataReadOnly(angZY));
    dev_centX    = static_cast<float const *>(mxGPUGetDataReadOnly(centX));
    dev_centY    = static_cast<float const *>(mxGPUGetDataReadOnly(centY));
    dev_elemXFst = static_cast<int const *>(mxGPUGetDataReadOnly(elemXFst));
    dev_elemXLst = static_cast<int const *>(mxGPUGetDataReadOnly(elemXLst));
    dev_elemYFst = static_cast<int const *>(mxGPUGetDataReadOnly(elemYFst));
    dev_elemYLst = static_cast<int const *>(mxGPUGetDataReadOnly(elemYLst));
    dev_nSampOmit= static_cast<int const *>(mxGPUGetDataReadOnly(nSampOmit));
    
    /* set constant memory */
    if(nElemX * nElemY != nRx) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "incompatible xElem, yElem, and iqRaw size");
    }
    if(nElemX > 256) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "xElem is too long, kernel supports xElem of up to 256 elements");
    }
    if(nElemY > 256) {
        mexErrMsgIdAndTxt(invalidInputMsgId, "yElem is too long, kernel supports yElem of up to 256 elements");
    }
    cudaMemcpyToSymbol(xElemConst, dev_xElem, nElemX*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(yElemConst, dev_yElem, nElemY*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    /* configure texture reference (apodization) */
    rxApodTex.normalized = true;
    rxApodTex.addressMode[0] = cudaAddressModeBorder;
    rxApodTex.filterMode = cudaFilterModeLinear;
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArrayApod;
    cudaMallocArray(&cuArrayApod, &channelDesc, nRxApodSamp, 0);
    cudaMemcpyToArray(cuArrayApod, 0, 0, dev_rxApod, nRxApodSamp*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaBindTextureToArray(rxApodTex, cuArrayApod, channelDesc);
    
    /* configure texture reference */
    iqRawTex.normalized  = false;
    iqRawTex.addressMode[0] = cudaAddressModeBorder;
    iqRawTex.filterMode  = cudaFilterModeLinear;
    
    int nTxPerPart = (nRx*nTx <= 2048) ? nTx : 2048/nRx;
    int nPart = (nTx+nTxPerPart-1)/nTxPerPart;
    
    cudaArray* cuArray;
    cudaExtent cuArraySize =  make_cudaExtent(nSamp, 0, nRx*nTxPerPart);
    cudaMalloc3DArray(&cuArray, &iqRawTex.channelDesc, cuArraySize, cudaArrayLayered);
    cudaBindTextureToArray(iqRawTex, cuArray);
    
    /* Kernel in loop - due to limited number of texture layers */
    cudaMemcpy3DParms cuArrayCopy = {0};
    cuArrayCopy.dstArray = cuArray;
    cuArrayCopy.kind = cudaMemcpyDeviceToDevice;
    for (int iPart=0; iPart<nPart; iPart++) {
        
        int nTxInThisPart = (iPart<(nPart-1)) ? nTxPerPart : (nTx-iPart*nTxPerPart);
        
        /* Prepare texture memory */
        cuArrayCopy.srcPtr = make_cudaPitchedPtr(const_cast<float2 *>(dev_iqRaw)+iPart*nSamp*nRx*nTxPerPart, nSamp * sizeof(float2), nSamp, 1);
        cuArrayCopy.extent = make_cudaExtent(nSamp, 1, nRx*nTxInThisPart);
        cudaMemcpy3D(&cuArrayCopy);
        
        /* Execute CUDA kernel */
        iqRaw2Lri<<<blocksPerGrid, threadsPerBlock, sharedPerBlock>>>(dev_iqLri, 
                                                                      dev_zPix, dev_xPix, dev_yPix, 
                                                                      dev_foc       + iPart*nTxPerPart, 
                                                                      dev_angZX     + iPart*nTxPerPart, 
                                                                      dev_angZY     + iPart*nTxPerPart, 
                                                                      dev_centX     + iPart*nTxPerPart, 
                                                                      dev_centY     + iPart*nTxPerPart, 
                                                                      dev_elemXFst  + iPart*nTxPerPart, 
                                                                      dev_elemXLst  + iPart*nTxPerPart, 
                                                                      dev_elemYFst  + iPart*nTxPerPart, 
                                                                      dev_elemYLst  + iPart*nTxPerPart, 
                                                                      dev_nSampOmit + iPart*nTxPerPart, 
                                                                      minRxTangZX, maxRxTangZX, 
                                                                      minRxTangZY, maxRxTangZY, 
                                                                      fs, fn, sos, initDel, 
                                                                      nZPix, nXPix, nYPix, 
                                                                      nSamp, nElemX, nElemY, 
                                                                      nTxInThisPart);
        
    }
    
    /* Wrap the output */
    plhs[0] = mxGPUCreateMxArrayOnGPU(iqLri);
    
    /* Clean-up */
    cudaUnbindTexture(iqRawTex);
    cudaFreeArray(cuArray);
    
    cudaUnbindTexture(rxApodTex);
    cudaFreeArray(cuArrayApod);
    
    mxGPUDestroyGPUArray(iqLri);
    mxGPUDestroyGPUArray(iqRaw);
    mxGPUDestroyGPUArray(xElem);
    mxGPUDestroyGPUArray(yElem);
    mxGPUDestroyGPUArray(zPix);
    mxGPUDestroyGPUArray(xPix);
    mxGPUDestroyGPUArray(yPix);
    mxGPUDestroyGPUArray(rxApod);
    mxGPUDestroyGPUArray(foc);
    mxGPUDestroyGPUArray(angZX);
    mxGPUDestroyGPUArray(angZY);
    mxGPUDestroyGPUArray(centX);
    mxGPUDestroyGPUArray(centY);
    mxGPUDestroyGPUArray(elemXFst);
    mxGPUDestroyGPUArray(elemXLst);
    mxGPUDestroyGPUArray(elemYFst);
    mxGPUDestroyGPUArray(elemYLst);
    mxGPUDestroyGPUArray(nSampOmit);
    
    //cudaDeviceReset();
}
