import cupy as cp

# TODO currently only complex input data is supported (part of DDC)

_gpu_fir_complex64_str = r'''
#include <cupy/complex.cuh>

extern "C" __global__ void gpu_fir_complex64(
    complex<float>* __restrict__ output, const complex<float>* __restrict__ input, 
    const int nSamples, const int length, const float* __restrict__ kernel, const int kernelWidth) {
    
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ch = idx / nSamples;
    int sample = idx % nSamples;
    
    extern __shared__ char sharedMemory[];
    
    complex<float>* cachedInputData = (complex<float>*)sharedMemory;
    float* cachedKernel = (float*)(sharedMemory+(kernelWidth+blockDim.x)*sizeof(complex<float>));
    
    
    // Cache kernel.
    for(int i = threadIdx.x;  i < kernelWidth; i += blockDim.x) {
        cachedKernel[i] = kernel[i];
    }        
    
    // Cache input.
    for(int i = sample-kernelWidth/2-1, localIdx = threadIdx.x; 
        localIdx < (kernelWidth+blockDim.x); i += blockDim.x, localIdx += blockDim.x) {
        if (i < 0 || i >= nSamples) {
            cachedInputData[localIdx] = 0;
        }
        else {
            cachedInputData[localIdx] = input[ch*nSamples + i];
        }
        
    }

    __syncthreads();

    if(idx >= length) {
        return;
    }
    complex<float> result(0.0f, 0.0f);
    int localN = threadIdx.x + kernelWidth;
    for (int i = 0; i < kernelWidth; ++i) {
        result += cachedInputData[localN-i]*cachedKernel[i];
    }
    output[idx] = result;
}
'''

gpu_fir_complex64 = cp.RawKernel(_gpu_fir_complex64_str, "gpu_fir_complex64")

_DEFAULT_BLOCK_SIZE = 512

def get_default_grid_block_size(n_samples, total_n_samples):
    block_size = (min((n_samples, _DEFAULT_BLOCK_SIZE)), )
    grid_size = (int((total_n_samples-1) // block_size[0] + 1), )
    return (grid_size, block_size)


def get_default_shared_mem_size(n_samples, filter_size):
    #  filter size (actual filter coefficients) + (block size + filter_size (padding))
    return filter_size*4 + (min(n_samples, _DEFAULT_BLOCK_SIZE) + filter_size)*8


def run_fir(grid_size, block_size, params, shared_mem_size):
    """
    :param params: a list: data_out, data_in, input_n_samples, input_total_n_samples, kernel, kernel_size
    """
    return gpu_fir_complex64(grid_size, block_size, params, shared_mem=shared_mem_size)




