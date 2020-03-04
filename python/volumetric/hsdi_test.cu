#include "hsdi.cuh"

void testPadWithHalfZeros() {
    double *dst, *src;
    cudaMalloc(&dst, DATA_SIZE*sizeof(dtype));
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
}


int main() {
    testPadWithHalfZeros();
}
