#include <iostream>
#include <fstream>
#include <array>
#include <cuda_runtime.h>
#include <cufft.h>

#include "hsdi.cuh"
#include "helper.h"

// SHAPE (NEVENTS, NCHANNELS_OX, NCHANNELS_OY, NSAMPLES)

const unsigned NEVENTS = 1;
const unsigned NCHANNELS_OX = 32;
const unsigned NCHANNELS_OY = 32;
const unsigned NSAMPLES = 2048;
const unsigned DATA_SIZE = NEVENTS*NCHANNELS_OX*NCHANNELS_OY*NSAMPLES;
std::array<dtype, DATA_SIZE> inputBuffer;

const unsigned NDEPTH = 1024;

const unsigned OUTPUT_SIZE = NCHANNELS_OX*NCHANNELS_OY*NDEPTH;
std::array<dtype, OUTPUT_SIZE> outputBuffer;

int main(int argc, char* argv[])
{
    typedef double realType;
    typedef cufftDoubleComplex complexType;
    // Read data
    realType *devInBuffer, *devPaddedBuffer, *devOutputBuffer;
    complexType *fftBuffer, *devInterpBuffer, *devIfftBuffer;
    cufftHandle fftPlanFwd, fftPlanInv;

    // ========================================== PREPARE
    const unsigned PADDED_OX = NCHANNELS_OX*2;
    const unsigned PADDED_OY = NCHANNELS_OY*2;


    const dtype SPEED_OF_SOUND = 1540;
    const dtype SAMPLING_FREQ = 25e6;
    const dtype PITCH = 0.3e-3;
    const dtype DF = SAMPLING_FREQ/NSAMPLES;
    // Porzadek jest zgodny z numpy (0, ... n/2-1, -n/2 .itd)
    const dtype DKX = 2*M_PI/(PADDED_OX*PITCH);
    const dtype DKY = 2*M_PI/(PADDED_OY*PITCH);
    const dtype F_MAX = SAMPLING_FREQ/2.0;
    const dtype KZ_MAX = 2*M_PI*F_MAX/SPEED_OF_SOUND;
    const dtype DKZ = KZ_MAX/(NDEPTH-1);

    std::ifstream input{"data.bin", std::ios::binary};
    input.read((char*)(inputBuffer.data()), inputBuffer.size()*sizeof(dtype));
    checkCudaErrors(cudaMalloc(&devInBuffer, DATA_SIZE*sizeof(dtype)));
    checkCudaErrors(cudaMemcpy(devInBuffer, inputBuffer.data(),
                               DATA_SIZE*sizeof(dtype), cudaMemcpyHostToDevice));

    const unsigned INPUT_PADDED_SIZE = NEVENTS*PADDED_OX*PADDED_OY*NSAMPLES;
    checkCudaErrors(cudaMalloc(&devPaddedBuffer, INPUT_PADDED_SIZE*sizeof(dtype)));

    const unsigned FFT_SIZE = INPUT_PADDED_SIZE/2+1;
    checkCudaErrors(cudaMalloc(&fftBuffer,
                               FFT_SIZE*sizeof(complexType)));
    const unsigned INTERP_SIZE = PADDED_OX*PADDED_OY*NDEPTH;
    checkCudaErrors(cudaMalloc(&devInterpBuffer,
                               INTERP_SIZE*sizeof(complexType)));
    checkCudaErrors(cudaMalloc(&devIfftBuffer,
                               INTERP_SIZE*sizeof(complexType)));
    checkCudaErrors(cudaMalloc(&devOutputBuffer,
                               OUTPUT_SIZE*sizeof(dtype)));

    // Cufft plans.
    checkCudaErrors(cufftPlan3d(&fftPlanFwd, PADDED_OX, PADDED_OY,
                                NSAMPLES, CUFFT_D2Z));
    checkCudaErrors(cufftPlan3d(&fftPlanInv, PADDED_OX, PADDED_OY,
                                NDEPTH, CUFFT_Z2Z));

    // ========================================== PROCESS
    // Pad with zeros.
    dim3 threads(32, 8, 1);
    dim3 grid(divup(PADDED_OX, threads.x),
              divup(PADDED_OY, threads.y),
              divup(NSAMPLES, threads.z));

    std::cout << "Padding with zeros" << std::endl;
    padHalfWithZeros<<<grid, threads>>>(devPaddedBuffer, devInBuffer,
                                        PADDED_OX, PADDED_OY,
                                        NCHANNELS_OX, NCHANNELS_OY, NSAMPLES);

    // FFT
    checkCudaErrors(cufftExecD2Z(fftPlanFwd, devPaddedBuffer, fftBuffer));

    // Interpolation & weighting
    dim3 threadsInterp(32, 8, 1);
    dim3 gridInterp(divup(PADDED_OX, threadsInterp.x),
                    divup(PADDED_OY, threadsInterp.y),
                    divup(NDEPTH, threadsInterp.z));
    interpWeight<<<gridInterp, threadsInterp>>>(
                 devInterpBuffer, fftBuffer,
                 PADDED_OX, PADDED_OY, NDEPTH,
                 PADDED_OX, PADDED_OY, NSAMPLES/2+1, // R2C was used
                 SPEED_OF_SOUND,
                 DKX, DKY, DKZ,
                 DF);

    // IFFT
    checkCudaErrors(cufftExecZ2Z(fftPlanInv, devInterpBuffer,
                                 devIfftBuffer, CUFFT_INVERSE));

    // remove padding, compute absolute value 
    unpadAbs<<<gridInterp, threadsInterp>>>(devOutputBuffer, devIfftBuffer,
                                           NCHANNELS_OX, NCHANNELS_OY,
                                           PADDED_OX, PADDED_OY, NDEPTH);

    // TODO(pjarosik) Abs, norm?
    // ========================================== DISPLAY OUTPUT
    checkCudaErrors(cudaMemcpy(outputBuffer.data(), devOutputBuffer,
                               OUTPUT_SIZE*sizeof(realType),
                               cudaMemcpyDeviceToHost));
    // Write output to a file.
    std::ofstream output{"pdata.bin", std::ios::binary};

    output.write((char*)(outputBuffer.data()), outputBuffer.size()*sizeof(realType));
    cudaFree(devInBuffer);
    cudaFree(devPaddedBuffer);
    cudaFree(devInterpBuffer);
    cudaFree(devIfftBuffer);
    cudaFree(devOutputBuffer);
    cufftDestroy(fftPlanFwd);
    cufftDestroy(fftPlanInv);
    return 0;
}
