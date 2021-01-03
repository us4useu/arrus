import cupy as cp

_arrus_remap_str = r'''
    // Naive implementation of data remapping (physical -> logical order).
    extern "C" 
    __global__ void arrus_remap(short* out, short* in, 
                                const short* fcmFrames, 
                                const char* fcmChannels, 
                                const unsigned nFrames, const unsigned nSamples, const unsigned nChannels)
    {
        int x = blockIdx.x * 32 + threadIdx.x; // logical channel
        int y = blockIdx.y * 32 + threadIdx.y; // logical sample
        int z = blockIdx.z; // logical frame
        if(x >= nChannels || y >= nSamples || z >= nFrames) {
            // outside the range
            return;
        }
        int indexOut = x + y*nChannels + z*nChannels*nSamples;
        int physicalChannel = fcmChannels[x + nChannels*z];
        if(physicalChannel < 0) {
            // channel is turned off
            return;
        }
        int physicalFrame = fcmFrames[x + nChannels*z];
        // 32 - number of channels in the physical mapping
        int indexIn = physicalChannel + y*32 + physicalFrame*32*nSamples; 
        out[indexOut] = in[indexIn];
    }'''


remap_kernel = cp.RawKernel(_arrus_remap_str, "arrus_remap")


def get_default_grid_block_size(fcm_frames, n_samples):
    # Note the kernel implementation
    block_size = (32, 32)
    n_frames, n_channels = fcm_frames.shape
    grid_size = (int((n_channels - 1) // block_size[0] + 1), int((n_samples - 1) // block_size[1] + 1), n_frames)
    return (grid_size, block_size)


def run_remap(grid_size, block_size, params):
    """
    :param params: a list: data_out, data_in, fcm_frames, fcm_channels, n_frames, n_samples, n_channels
    """
    return remap_kernel(grid_size, block_size, params)

