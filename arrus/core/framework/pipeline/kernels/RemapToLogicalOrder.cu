#ifndef CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH
#define CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH

#include "imaging/CudaUtils.cuh"
#include "imaging/kernels/RemapToLogicalOrder.h"

namespace arrus_example_imaging {

/**
 * Generic implementation of remapping the input data from us4OEM specific order (determined by the us4OEM
 * system architecture) to logical order, specific to TX/RX parameters set by user.
 * Due to its genericity, this kernel is not necessarily the most optimal way to implement the remapping.
 *
 * In the below documentation:
 * - the 'global number of frame' refers to the number of frame within all physical or logical frames,
 * - the 'local number of frame' refers to the number of frame within all frames of a single sequence,
 *
 * @param out output array
 * @param in input array
 * @param fcmFrames FCM frame numbers, shape: (n logical frames, n logical channels)
 * @param fcmChannels FCM channel numbers, shape (n logical frames, n logical channels)
 * @param fcmUs4oems FCM us4OEM numbers, shape (n logical frames, n logical channels)
 * @param frameOffsets FCM frame offsets for each us4OEM. The global number of the first frame, acquired by this us4OEM.
 * @param nFramesUs4OEM number of physical frames within a single sequence, for each us4OEM
 * @param nSequences number of sequences per batch (output shape)
 * @param nFrames number of frames per sequence (output shape)
 * @param nSamples number of samples per frame (output shape)
 * @param nChannels number of channels per frame (output shape)
 */
__global__ void gpuUs4rRemap(short *out, const short *in, const short *fcmFrames, const char *fcmChannels,
                              const unsigned char *fcmUs4oems, const unsigned int *frameOffsets,
                              const unsigned int *nFramesUs4OEM, const unsigned nSequences, const unsigned nFrames,
                              const unsigned nSamples, const unsigned nChannels) {
    int channel = blockIdx.x * RemapToLogicalOrderFunctor::BLOCK_TILE_DIM + threadIdx.x;// logical channel
    int sample = blockIdx.y * RemapToLogicalOrderFunctor::BLOCK_TILE_DIM + threadIdx.y; // logical sample
    int frame = blockIdx.z;// logical frame, global in the whole batch of sequences
    // Determine sequence number (in batch) and frame number (within sequence)
    int sequence = frame / nFrames;
    int localFrame = frame % nFrames;

    if (channel >= nChannels || sample >= nSamples || localFrame >= nFrames || sequence >= nSequences) {
        // outside the range
        return;
    }
    // [sequence, frame, sample, channel]
    size_t indexOut =
        sequence * nFrames * nSamples * nChannels + localFrame * nSamples * nChannels + sample * nChannels + channel;

    int physicalChannel = fcmChannels[channel + nChannels * localFrame];
    if (physicalChannel < 0) {
        // channel is turned off
        return;
    }
    int physicalFrame = fcmFrames[channel + nChannels * localFrame];
    int us4oem = fcmUs4oems[channel + nChannels * localFrame];
    int us4oemOffset = frameOffsets[us4oem];
    int nPhysicalFrames = nFramesUs4OEM[us4oem];
    const int nRx = 32;
    // 32 = number of RX channels a single us4OEM acquires. Note: should be parametrized (e.g. 64 channels may be used
    // in the future).
    // [us4oem, sequence, physicalFrame, sample, physicalChannel]
    size_t indexIn = us4oemOffset * nSamples * nRx + sequence * nPhysicalFrames * nSamples * nRx
        + physicalFrame * nSamples * nRx + sample * nRx + physicalChannel;
    out[indexOut] = in[indexIn];
}

/**
 * @param nSequences number of frames in a single logical sequence
 * @param nFrames number of local logical frames
 * @param nChannels number of logical channels
 * @param nSamples number of samples per logical frame
 */
void RemapToLogicalOrderFunctor::operator()(NdArray &output, const NdArray &input, const NdArray &fcmFrames,
                                            const NdArray &fcmChannels, const NdArray &fcmUs4oems,
                                            const NdArray &frameOffsets, const NdArray &nFramesUs4OEM,
                                            const unsigned nSequences, const unsigned nFrames,
                                            const unsigned nSamples, const unsigned nChannels,
                                            cudaStream_t stream) {
    dim3 block(BLOCK_TILE_DIM, BLOCK_TILE_DIM);
    dim3 grid((nChannels - 1) / block.x + 1, (nSamples - 1) / block.y + 1, nFrames*nSequences);
    gpuUs4rRemap<<<grid, block, 0, stream>>>(
        output.getPtr<int16_t>(), input.getConstPtr<int16_t>(), fcmFrames.getConstPtr<int16_t>(),
        fcmChannels.getConstPtr<char>(), fcmUs4oems.getConstPtr<uint8_t>(), frameOffsets.getConstPtr<uint32_t>(),
        nFramesUs4OEM.getConstPtr<uint32_t>(), nSequences, nFrames, nSamples, nChannels);
    CUDA_ASSERT(cudaGetLastError());
}

}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH
