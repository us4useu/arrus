"""
Cupy implementation of the scipy.interpolate.interp1d function.
The gpu function is currently very limited - it allows only for linear
interpolation, with constant value (equal 0.0) when extrapolating data.
"""
import cupy as cp


# TODO(pjarosik) move to .cuh
_interp1d_kernel_str = r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void interp1d_kernel_%%dtype_name%%(
            const %%dtype%%* __restrict__ data, const size_t dataWidth, const size_t dataHeight, 
            const float* __restrict__ samples, 
            %%dtype%%* __restrict__ output, const size_t outputWidth, const size_t outputHeight){

        int xt = blockDim.x * blockIdx.x + threadIdx.x;

        if(xt >= outputWidth) {
            return;
        }
        float samplePos = samples[xt]; 
        int sampleNr = floorf(samplePos);
        float ratio = samplePos - sampleNr; 

        for(int i = 0; i < outputHeight; ++i) {
            size_t outputOffset = i*outputWidth;
            size_t dataOffset = i*dataWidth;

            if(   (sampleNr < 0) 
               || (sampleNr >= dataWidth) 
               || (sampleNr == dataWidth-1 && ratio != 0.0f)) { // right border + epsilon
                // extrapolation
                output[xt+outputOffset] = 0;
            }
            else {
                // interpolation
                output[xt+outputOffset] = (1-ratio)*data[sampleNr+dataOffset] 
                                     + ratio*data[sampleNr+1+dataOffset];
            }
        }
    }'''

_interp1d_kernel_complex64 = cp.RawKernel(_interp1d_kernel_str
                                          .replace("%%dtype%%",
                                                   "complex<float>")
                                          .replace("%%dtype_name%%",
                                                   "complex64"),
                                          "interp1d_kernel_complex64")
_interp1d_kernel_float32 = cp.RawKernel(_interp1d_kernel_str
                                        .replace("%%dtype%%", "float")
                                        .replace("%%dtype_name%%", "float32"),
                                        "interp1d_kernel_float32")


def interp1d(input_data, samples, output_data):
    samples = samples.squeeze()
    if samples.ndim > 1:
        raise ValueError("'samples' should be a 1D vector.")
    blockSize = (512,)
    output_height, output_width = output_data.shape
    input_height, input_width = input_data.shape
    gridSize = (int((output_width - 1) // blockSize[0] + 1),)
    params = (input_data, input_width, input_height,
              samples,
              output_data, output_width, output_height)
    if input_data.dtype == cp.complex64:
        return _interp1d_kernel_complex64(gridSize, blockSize, params)
    elif input_data.dtype == cp.float32:
        return _interp1d_kernel_float32(gridSize, blockSize, params)
    else:
        raise ValueError(f"Unsupported data type: {input_data.dtype}")
