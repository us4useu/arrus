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
const unsigned PADDED_DATA_SIZE = NEVENTS*PADDED_OX*PADDED_OY*NSAMPLES;
std::array<dtype, PADDED_DATA_SIZE> outputBuffer;

int main(int argc, char* argv[])
{
    typedef double realType;
    typedef cufftDoubleComplex complexType;
    // Read data
    realType *devInBuffer, *devProcBuffer, *devOutputBuffer;
    complexType *fftBuffer;
    cufftHandle fftPlanFwd, fftPlanInv;

    std::ifstream input{"data.bin", std::ios::binary};
    input.read((char*)(inputBuffer.data()), inputBuffer.size()*sizeof(dtype));
    checkCudaErrors(cudaMalloc(&devInBuffer, DATA_SIZE*sizeof(dtype)));
    checkCudaErrors(cudaMemcpy(devInBuffer, inputBuffer.data(), DATA_SIZE*sizeof(dtype), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&devProcBuffer, PADDED_DATA_SIZE*sizeof(dtype)));
    checkCudaErrors(cudaMalloc(&fftBuffer,
                               (PADDED_DATA_SIZE/2+1)*sizeof(complexType)));
    checkCudaErrors(cudaMalloc(&devOutputBuffer,
                               (PADDED_DATA_SIZE)*sizeof(realType)));
    checkCudaErrors(cufftPlan3d(&fftPlanFwd, PADDED_OX, PADDED_OY,
                                NSAMPLES, CUFFT_D2Z));
    checkCudaErrors(cufftPlan3d(&fftPlanInv, PADDED_OX, PADDED_OY,
                                NSAMPLES, CUFFT_Z2D));


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

    // IFFT
    checkCudaErrors(cufftExecZ2D(fftPlanInv, fftBuffer, devOutputBuffer));

    // Abs, norm?

    // Write output to a file.
    std::cout << "Producing the output, size: "
              << PADDED_DATA_SIZE*sizeof(dtype)
              << std::endl;
    std::ofstream output{"pdata.bin", std::ios::binary};

    checkCudaErrors(cudaMemcpy(outputBuffer.data(), devProcBuffer,
                               PADDED_DATA_SIZE*sizeof(dtype),
                               cudaMemcpyDeviceToHost));

    output.write((char*)(outputBuffer.data()), outputBuffer.size()*sizeof(dtype));
    cudaFree(devInBuffer);
    cudaFree(devProcBuffer);
    cufftDestroy(fftPlanFwd);
    cufftDestroy(fftPlanInv);
    return 0;
}
