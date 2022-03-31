import cupy as cp

# TODO strategy for case batch size == 1


_arrus_remap_str = r'''
    // Naive implementation of data remapping (physical -> logical order).
    extern "C" 
    __global__ void arrus_remap(short* out, short* in, 
                                const short* fcmFrames, 
                                const char* fcmChannels, 
                                const unsigned char *fcmUs4oems,
                                // Number of sample, that starts given us4oEM data
                                const unsigned int *frameOffsets,
                                const unsigned int *nFramesUs4OEM,
                                // Output shape
                                const unsigned nSequences, const unsigned nFrames, 
                                const unsigned nSamples, const unsigned nChannels)
    {
        int channel = blockIdx.x*32 + threadIdx.x; // logical channel
        int sample = blockIdx.y*32 + threadIdx.y; // logical sample
        int frame = blockIdx.z; // logical frame, global in the whole batch of sequences
        // Determine sequence number (in batch) and frame number (within sequence)
        int sequence = frame / nFrames;
        int localFrame = frame % nFrames;
        
        if(channel >= nChannels || sample >= nSamples || localFrame >= nFrames || sequence >= nSequences) {
            // outside the range
            return;
        }
        
        // FCM describes here a single sequence 
        int physicalChannel = fcmChannels[channel + nChannels*localFrame];
        if(physicalChannel < 0) {
            // channel is turned off
            return;
        }
        // [sequence, frame, sample, channel]
        size_t indexOut = sequence*nFrames*nSamples*nChannels 
                        + localFrame*nSamples*nChannels
                        + sample*nChannels 
                        + channel;
            
        // FCM describes here a single sequence
        int physicalFrame = fcmFrames[channel + nChannels*localFrame];
        // 32 - number of channels in the physical mapping
        // [us4oem, sequence, physicalFrame, sample, physicalChannel]
        
        int us4oem = fcmUs4oems[channel + nChannels*localFrame];
        int us4oemOffset = frameOffsets[us4oem];
        int nPhysicalFrames = nFramesUs4OEM[us4oem];
        
        size_t indexIn = us4oemOffset*nSamples*32
                       + sequence*nPhysicalFrames*nSamples*32
                       + physicalFrame*nSamples*32 
                       + sample*32 
                       + physicalChannel; 
        out[indexOut] = in[indexIn];
    }'''


remap_kernel = cp.RawKernel(_arrus_remap_str, "arrus_remap")


def get_default_grid_block_size(fcm_frames, n_samples, batch_size):
    # Note the kernel implementation
    block_size = (32, 32)
    n_frames, n_channels = fcm_frames.shape
    grid_size = (
        (n_channels - 1) // block_size[0] + 1,
        (n_samples - 1) // block_size[1] + 1,
        n_frames*batch_size
    )
    return (grid_size, block_size)


def run_remap(grid_size, block_size, params):
    """
    :param params: a list: data_out, data_in, fcm_frames, fcm_channels, n_frames, n_samples, n_channels
    """
    return remap_kernel(grid_size, block_size, params)

