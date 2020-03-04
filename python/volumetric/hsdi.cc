#include <iostream>
#include <fstream>
#include <array>

const unsigned NEVENTS = 1;
const unsigned NCHANNELS_OX = 32;
const unsigned NCHANNELS_OY = 32;
const unsigned NSAMPLES = 2048;
const unsigned DATA_SIZE = NEVENTS*NCHANNELS_OX*NCHANNELS_OY*NSAMPLES;

// SHAPE (NEVENTS, NCHANNELS_OX, NCHANNELS_OY, NSAMPLES)

typedef double dtype;

std::array<dtype, DATA_SIZE> buffer;

int main(int argc, char* argv[]) {
    // Read data
    std::ifstream input("data.bin", std::ios::binary);

    input.read((char*)(buffer.data()), buffer.size()*sizeof(dtype));
    
    std::ofstream output("pdata.bin", std::ios::binary);
    output.write((char*)(buffer.data()), buffer.size()*sizeof(dtype));

    return 0;
}
