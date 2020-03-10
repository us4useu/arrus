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

const unsigned PADDED_OX = NCHANNELS_OX*2;
const unsigned PADDED_OY = NCHANNELS_OY*2;
const unsigned NDEPTH = 1024;
const unsigned PADDED_DATA_SIZE = NEVENTS*PADDED_OX*PADDED_OY*NDEPTH;
std::array<complexType, PADDED_DATA_SIZE> outputBuffer;

int main(int argc, char* argv[])
{
    typedef double realType;
    typedef cufftDoubleComplex complexType;
    // Read data
    realType *devInBuffer, *devProcBuffer;
    complexType *fftBuffer, *devOutputBuffer;
    cufftHandle fftPlanFwd, fftPlanInv;

    // Precomputed parameters.
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
    checkCudaErrors(cudaMemcpy(devInBuffer, inputBuffer.data(), DATA_SIZE*sizeof(dtype), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&devProcBuffer, PADDED_DATA_SIZE*sizeof(dtype)));
    checkCudaErrors(cudaMalloc(&fftBuffer,
                               (PADDED_DATA_SIZE/2+1)*sizeof(complexType)));
    checkCudaErrors(cudaMalloc(&devOutputBuffer,
                               (PADDED_DATA_SIZE)*sizeof(complexType)));
    checkCudaErrors(cufftPlan3d(&fftPlanFwd, PADDED_OX, PADDED_OY,
                                NSAMPLES, CUFFT_D2Z));
    checkCudaErrors(cufftPlan3d(&fftPlanInv, PADDED_OX, PADDED_OY,
                                NSAMPLES, CUFFT_Z2Z));


    // Pad with zeros.
    dim3 threads(32, 8, 1);
    dim3 grid(divup(PADDED_OX, threads.x),
              divup(PADDED_OY, threads.y),
              divup(NSAMPLES, threads.z));

    std::cout << "Padding with zeros" << std::endl;
    padHalfWithZeros<<<grid, threads>>>(devProcBuffer, devInBuffer,
                                        PADDED_OX, PADDED_OY,
                                        NCHANNELS_OX, NCHANNELS_OY, NSAMPLES);
    // FFT
    checkCudaErrors(cufftExecD2Z(fftPlanFwd, devProcBuffer, fftBuffer));

    // Interpolation & weighting
    interpWeight<<<grid, threads>>>(devOutputBuffer, fftBuffer,
                 PADDED_OX, PADDED_OY, NDEPTH,
                 PADDED_OX, PADDED_OY, NSAMPLES/2+1, // R2C was used
                 SPEED_OF_SOUND,
                 DKX, DKY, DKZ,
                 DF);

    // IFFT
    checkCudaErrors(cufftExecZ2Z(fftPlanInv, fftBuffer, devOutputBuffer, CUFFT_INVERSE));

    // TODO(pjarosik) unpadd?

    // TODO(pjarosik) Abs, norm?

    // Write output to a file.
    std::ofstream output{"pdata.bin", std::ios::binary};

    checkCudaErrors(cudaMemcpy(outputBuffer.data(), devOutputBuffer,
                               PADDED_DATA_SIZE*sizeof(complexType),
                               cudaMemcpyDeviceToHost));

    output.write((char*)(outputBuffer.data()), outputBuffer.size()*sizeof(complexType));
    cudaFree(devInBuffer);
    cudaFree(devProcBuffer);
    cudaFree(devOutputBuffer);
    cufftDestroy(fftPlanFwd);
    cufftDestroy(fftPlanInv);
    return 0;
}
