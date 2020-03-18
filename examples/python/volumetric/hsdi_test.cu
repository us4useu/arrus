#include "hsdi.cuh"
#include <iostream>
#include <cassert>
#include <array>

void testPadWithHalfZeros() {

    // Create.
    // -- Input.
    dtype *dst, *src;
    const int srcW = 2;
    const int srcH = 2;
    const int depth = 2;
    
    cudaMallocManaged(&src, srcW*srcH*depth*sizeof(dtype));
    std::cout << "Input array: " << std::endl;

    for(int i = 0; i < srcW; ++i) {
        for(int j = 0; j < srcH; ++j) {
            for(int k = 0; k < depth; ++k) {
                int idx = k + depth*j + depth*srcH*i;
                src[idx] = idx+1;
                std::cout << src[idx] << ", ";
            }
        }
    }
    std::cout << std::endl;

    // -- Output.
    const int dstW = 4;
    const int dstH = 4;
    const int outputSize = dstW*dstH*depth*sizeof(dtype);
    cudaMallocManaged(&dst, dstW*dstH*depth*sizeof(dtype));

    dim3 threads(2, 2, 1);
    dim3 grid(divup(dstW, threads.x),
              divup(dstH, threads.y),
              divup(depth, threads.z));

    // Perform.
    padHalfWithZeros<<<grid, threads>>>(dst, src,
                                        dstW, dstH, srcW, srcH, depth);

    std::array<dtype, outputSize> output;
    cudaMemcpy(output.data(), dst, outputSize, cudaMemcpyDefault);

    // Validate.
    std::cout << "Computed array: " << std::endl;
    for(int i = 0; i < dstW*dstH*depth; ++i) {
        std::cout << output[i] << ", ";
    }
    std::cout << std::endl;

    for(int i = 0; i < dstW; ++i) {
        for(int j = 0; j < dstH; ++j) {
            for(int k = 0; k < depth; ++k) {
                int dstIdx = k + depth*j + depth*dstH*i;
                if(i >= 1 && i < 3 && j >= 1 && j < 3) {
                    // Assertion.
                    int srcIdx = k + depth*(j-1) + depth*srcH*(i-1);
                    assert(output[dstIdx] == src[srcIdx]);
                }
                else {
                    assert(output[dstIdx] == 0);
                }
            }
        }
    }
    std::cout << std::endl;
}


int main() {
    testPadWithHalfZeros();
    std::cout << "OK" << std::endl;
}
