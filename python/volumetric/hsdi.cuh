#include <cstdio>
#include <math.h>
#include <cufft.h>
#include <cuComplex.h>
#include "helper.h"

typedef double dtype;
typedef double realType;
typedef cufftDoubleComplex complexType;

inline unsigned divup(unsigned x, unsigned div) {
    return (x + div-1) / div;
}

__global__ void padHalfWithZeros(dtype* dst, dtype* src,
                                 int dstW, int dstH,
                                 int srcW, int srcH, int depth)
{
    const int x = blockDim.x*blockIdx.x + threadIdx.x;
    const int y = blockDim.y*blockIdx.y + threadIdx.y;
    const int z = blockDim.z*blockIdx.z + threadIdx.z;
    const int dstIdx = z + y*depth + x*depth*dstH;
    const int leftBorderX = (dstW-srcW)/2;
    const int rightBorderX = leftBorderX+srcW;

    const int leftBorderY = (dstH-srcH)/2;
    const int rightBorderY = leftBorderY+srcH;

    // TODO(pjarosik) grid-stride loop
    if(x < dstW && y < dstH && z < depth) {
        if (x >= leftBorderX && x < rightBorderX &&
            y >= leftBorderY && y < rightBorderY) {

            const int srcIdx = z+(y-leftBorderY)*depth+(x-leftBorderX)*depth*srcH;
            dst[dstIdx] = src[srcIdx];
        }
        else {
            dst[dstIdx] = 0.0;
        }
    }
}

// :param dkz: a change in a single dkz axis
__global__ void interpWeight(complexType* dst, complexType* src,
                             int dstW, int dstH, int dstD,
                             int srcW, int srcH, int srcD,
                             dtype speedOfSound,
                             dtype dkx, dtype dky, dtype dkz,
                             dtype df)
{
    const int x = blockDim.x*blockIdx.x + threadIdx.x;
    const int y = blockDim.y*blockIdx.y + threadIdx.y;
    const int z = blockDim.z*blockIdx.z + threadIdx.z;
    const int halfW = srcW/2;
    const int halfH = srcH/2;
    const int dstIdx = z + y*dstD + x*dstD*dstH;

    if(x < dstW && y < dstH && z < dstD) {
        if(z == 0) {
            dst[dstIdx] = {0.0, 0.0};
        }
        else {
            // TODO(pjarosik) consider moving to shared memory
            dtype kx = dkx*(-halfW+(x+halfW)%srcW);
            dtype ky = dky*(-halfH+(y+halfH)%srcH);
            dtype kz = dkz*z;

            dtype sample = speedOfSound/(4*M_PI)*((kz*kz+kx*kx+ky*ky)/kz);
            
            // Sample at df should be equal one (just cast frequencies to
            // index numbers).
            sample = sample/df;
            dtype w = speedOfSound/(4*M_PI)*((kz*kz-kx*kx-ky*ky)/kz);

            // linear interp.
            int z1 = int(sample);
            if(z1 >= srcD) {
                dst[dstIdx] = {0.0, 0.0};
                return;
            }
            int z2 = z1+1;
            if(z2 >= srcD) {
                dst[dstIdx] = src[z1+y*srcD+x*srcD*srcH];
                return;
            }
            dtype alpha = sample-z1;
            complexType y1 = src[z1+y*srcD+x*srcD*srcH];
            complexType y2 = src[z2+y*srcD+x*srcD*srcH];
            dst[dstIdx] = {w*((1-alpha)*y1.x+alpha*y2.x),
                           w*((1-alpha)*y1.y+alpha*y2.y)};
        }
    }
}

__global__ void unpadAbs(dtype* dst, complexType* src,
                         int dstW, int dstH,
                         int srcW, int srcH, int depth)
{
    const int x = blockDim.x*blockIdx.x + threadIdx.x;
    const int y = blockDim.y*blockIdx.y + threadIdx.y;
    const int z = blockDim.z*blockIdx.z + threadIdx.z;
    const int dstIdx = z + y*depth + x*depth*dstH;

    const int leftBorderX = (srcW-dstW)/2;
    const int leftBorderY = (srcH-dstH)/2;

    if(x < dstW && y < dstH && z < depth) {
        const int srcIdx = z+(y+leftBorderY)*depth+(x+leftBorderX)*depth*srcH;
        dst[dstIdx] = cuCabs(src[srcIdx]);
    }
}


class HSDIOp {
public:
    HSDIOp(const unsigned gpuIdx,
           const unsigned nChannelsOx,
           const unsigned nChannelsOy,
           const unsigned nSamples,
           const unsigned outputDepth,
           const unsigned padding=2
           ) {
        this->gpuIdx = gpuIdx;
        this->nChannelsOx = nChannelsOx;
        this->nChannelsOy = nChannelsOy;
        this->nSamples = nSamples;
        this->outputDepth = outputDepth;

        // Precomputing standard values.
        this->paddedOx = padding*nChannelsOx;
        this->paddedOy = padding*nChannelsOy;

        const dtype samplingFreq = 25e6;
        const dtype pitch = 0.3e-3;
        this->speedOfSound = 1540;

        this->df = samplingFreq/this->nSamples;
        this->dkx = 2*M_PI/(this->paddedOx*pitch);
        this->dky = 2*M_PI/(this->paddedOy*pitch);

        const dtype fMax = samplingFreq/2.0;
        const dtype kzMax = 2*M_PI*fMax/this->speedOfSound;
        this->dkz = kzMax/(this->outputDepth-1);


        // Allocating memory.
        this->inputSize = nChannelsOx*nChannelsOy*nSamples;
        const unsigned inputPaddedSize = paddedOx*paddedOy*nSamples;
        checkCudaErrors(cudaMalloc(&devInBuffer, this->inputSize*sizeof(dtype)));
        checkCudaErrors(cudaMalloc(&devPaddedBuffer,
                                   inputPaddedSize*sizeof(dtype)));
        const unsigned fftSize = inputPaddedSize/2+1;
        checkCudaErrors(cudaMalloc(&fftBuffer,
                                   fftSize*sizeof(complexType)));
        const unsigned interpSize = paddedOx*paddedOy*outputDepth;
        checkCudaErrors(cudaMalloc(&devInterpBuffer,
                                   interpSize*sizeof(complexType)));
        checkCudaErrors(cudaMalloc(&devIfftBuffer,
                                   interpSize*sizeof(complexType)));
        const unsigned outputSize = nChannelsOx*nChannelsOy*outputDepth;
        checkCudaErrors(cudaMalloc(&devOutputBuffer,
                                   outputSize*sizeof(dtype)));

        // Cufft plans
        checkCudaErrors(cufftPlan3d(&fftPlanFwd, paddedOx, paddedOy,
                                    nSamples, CUFFT_D2Z));
        checkCudaErrors(cufftPlan3d(&fftPlanInv, paddedOx, paddedOy,
                                    outputDepth, CUFFT_Z2Z));
    }

    void process(dtype* inputBuffer) {
        checkCudaErrors(cudaMemcpy(this->devInBuffer, inputBuffer,
                                   this->inputSize*sizeof(dtype),
                                   cudaMemcpyHostToDevice));
        // Pad with zeros.
        dim3 threads(32, 8, 1);
        dim3 grid(divup(this->paddedOx, threads.x),
                  divup(this->paddedOy, threads.y),
                  divup(this->nSamples, threads.z));
        padHalfWithZeros<<<grid, threads>>>(this->devPaddedBuffer,
                                            this->devInBuffer,
                                            this->paddedOx, this->paddedOy,
                                            this->nChannelsOx, this->nChannelsOy,
                                            this->nSamples);
        // FFT
        checkCudaErrors(cufftExecD2Z(fftPlanFwd,
                                     this->devPaddedBuffer,
                                     this->fftBuffer));

        // Interpolation & weighting
        dim3 threadsInterp(32, 8, 1);
        dim3 gridInterp(divup(this->paddedOx, threadsInterp.x),
                        divup(this->paddedOy, threadsInterp.y),
                        divup(this->outputDepth, threadsInterp.z));
        interpWeight<<<gridInterp, threadsInterp>>>(
                        this->devInterpBuffer,
                        this->fftBuffer,
                        this->paddedOx, this->paddedOy, this->outputDepth,
                        this->paddedOx, this->paddedOy, this->nSamples/2+1,
                        this->speedOfSound,
                        this->dkx, this->dky, this->dkz,
                        this->df);
        // IFFT
        checkCudaErrors(cufftExecZ2Z(fftPlanInv,
                                     this->devInterpBuffer,
                                     this->devIfftBuffer,
                                     CUFFT_INVERSE));
        // remove padding, compute absolute value 
        unpadAbs<<<gridInterp, threadsInterp>>>(this->devOutputBuffer,
                                                this->devIfftBuffer,
                                                this->nChannelsOx,
                                                this->nChannelsOy,
                                                this->paddedOx,
                                                this->paddedOy,
                                                this->outputDepth);
    }

    ~HSDIOp() {
        checkCudaErrors(cudaFree(devInBuffer));
        checkCudaErrors(cudaFree(devPaddedBuffer));
        checkCudaErrors(cudaFree(devInterpBuffer));
        checkCudaErrors(cudaFree(devIfftBuffer));
        checkCudaErrors(cudaFree(devOutputBuffer));
        checkCudaErrors(cufftDestroy(fftPlanFwd));
        checkCudaErrors(cufftDestroy(fftPlanInv));
    }

    realType* getOutput() {
        return this->devOutputBuffer;
    }

private:
    unsigned gpuIdx;
    realType *devInBuffer, *devPaddedBuffer, *devOutputBuffer;
    complexType *fftBuffer, *devInterpBuffer, *devIfftBuffer;
    cufftHandle fftPlanFwd, fftPlanInv;
    unsigned nChannelsOx, nChannelsOy, nSamples, outputDepth, padding,
        paddedOx, paddedOy;
    unsigned inputSize;
    dtype speedOfSound, dkx, dky, dkz, df;
};
