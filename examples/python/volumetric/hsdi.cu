#include <iostream>
#include <fstream>
#include <array>
#include <cuda_runtime.h>
#include <cufft.h>
#include "hsdi.cuh"

int main(int argc, char* argv[])
{
    if(argc != 6) {
        std::cerr << "Invalid number of input arguments" << std::endl;
        return -1;
    }
    const int nEvents = atoi(argv[1]);
    const unsigned nChannelsOx = atoi(argv[2]);
    const unsigned nChannelsOy = atoi(argv[3]);
    const unsigned nSamples = atoi(argv[4]);
    const unsigned outputDepth = atoi(argv[5]);
    
    const unsigned dataSize = nEvents*nChannelsOx*nChannelsOy*nSamples;
    const unsigned outputSize = nChannelsOx*nChannelsOy*outputDepth;
    dtype* inputBuffer = new dtype[dataSize];
    dtype* outputBuffer = new dtype[outputSize];
    std::ifstream input{"data.bin", std::ios::binary};
    input.read((char*)(inputBuffer), dataSize*sizeof(dtype));

    HSDIOp op = HSDIOp(0, nChannelsOx, nChannelsOy, nSamples, outputDepth);
    op.process(inputBuffer);

    checkCudaErrors(cudaMemcpy(outputBuffer, op.getOutput(),
                               outputSize*sizeof(realType),
                               cudaMemcpyDeviceToHost));
    // Write output to a file.
    std::ofstream output{"pdata.bin", std::ios::binary};
    output.write((char*)(outputBuffer), outputSize*sizeof(realType));
    delete[] inputBuffer;
    delete[] outputBuffer;
    return 0;
}
