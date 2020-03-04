#include <iostream>
#include <fstream>
#include <array>
#include "hsdi.cuh"


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
    // Read data
    std::cout << "Reading the data." << std::endl;
    double* devInBuffer = 0;
    std::ifstream input{"data.bin", std::ios::binary};

    input.read((char*)(inputBuffer.data()), inputBuffer.size()*sizeof(dtype));
    cudaMalloc(&devInBuffer, DATA_SIZE*sizeof(dtype));
    cudaMemcpy(&devInBuffer, inputBuffer.data(), DATA_SIZE*sizeof(dtype),
               cudaMemcpyHostToDevice);

    // Padd data with zeros.
    double* devProcBuffer = 0;
    cudaMalloc(&devProcBuffer, PADDED_DATA_SIZE*sizeof(dtype));

    dim3 threads(32, 8, 1);
    dim3 grid(divup(PADDED_OX, threads.x),
              divup(PADDED_OY, threads.y),
              divup(NSAMPLES, threads.z));

    std::cout << "Padding with zeros" << std::endl;
    padHalfWithZeros<<<grid, threads>>>(devProcBuffer, devInBuffer,
                                        PADDED_OX, PADDED_OY,
                                        NCHANNELS_OX, NCHANNELS_OY, NSAMPLES);

    // Produce output.
    std::cout << "Producing the output, size: "
              << PADDED_DATA_SIZE*sizeof(dtype)
              << std::endl;
    std::ofstream output{"pdata.bin", std::ios::binary};

    cudaMemcpy(&outputBuffer, devProcBuffer, PADDED_DATA_SIZE*sizeof(dtype),
               cudaMemcpyDeviceToHost);

    output.write((char*)(outputBuffer.data()), outputBuffer.size()*sizeof(dtype));
    cudaFree(devInBuffer);
    cudaFree(devProcBuffer);
    return 0;
}
