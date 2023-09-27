/**
 * Naive implementation of data remapping (physical -> logical order).
 *
 * @param out: output array
 * @param in: input array
 * @param fcmFrames: frame channel mapping: frames
 * @param fcmChannels: frame channel mapping: channels
 * @param fcmUs4oems: frame channel mapping: us4oems
 * @parma frameOffsets: Number of frames, that starts given us4OEM data
 * @param nFramesUs4OEM: number of frames each us4OEM acquires
 * @param nSequences, nFrames, nSamples, nChannels: output shape
 */
extern "C" __global__ void arrusRemap(short *out, short *in, const short *fcmFrames, const char *fcmChannels,
                                       const unsigned char *fcmUs4oems, const unsigned int *frameOffsets,
                                       const unsigned int *nFramesUs4OEM, const unsigned nSequences,
                                       const unsigned nFrames, const unsigned nSamples, const unsigned nChannels) {
    int channel = blockIdx.x * 32 + threadIdx.x;// logical channel
    int sample = blockIdx.y * 32 + threadIdx.y; // logical sample
    int frame = blockIdx.z;                     // logical frame, global in the whole batch of sequences

    int sequence = frame / nFrames;
    int localFrame = frame % nFrames;
    if (channel >= nChannels || sample >= nSamples || localFrame >= nFrames || sequence >= nSequences) {
        return;
    }
    // FCM describes here a single sequence
    int physicalChannel = fcmChannels[channel + nChannels*localFrame];
    if (physicalChannel < 0) {
        // channel is turned off
        return;
    }
    // [sequence, frame, sample, channel]
    size_t indexOut =
        sequence*nFrames*nSamples*nChannels + localFrame*nSamples*nChannels + sample*nChannels + channel;

    // 32 == number of channels in the physical mapping
    // [us4oem, sequence, physicalFrame, sample, physicalChannel]
    int physicalFrame = fcmFrames[channel + nChannels*localFrame];
    int us4oem = fcmUs4oems[channel + nChannels*localFrame];
    int us4oemOffset = frameOffsets[us4oem];
    int nPhysicalFrames = nFramesUs4OEM[us4oem];

    size_t indexIn = us4oemOffset*nSamples*32 + sequence*nPhysicalFrames*nSamples*32
        + physicalFrame*nSamples*32 + sample*32 + physicalChannel;
    out[indexOut] = in[indexIn];
}

/**
 * Naive implementation of data remapping (physical -> logical order), version 2.
 *
 *
 * @param out: output array
 * @param in: input array
 * @param fcmFrames: frame channel mapping: frames
 * @param fcmChannels: frame channel mapping: channels
 * @param fcmUs4oems: frame channel mapping: us4oems
 * @parma frameOffsets: Number of frame (global), that starts given us4OEM data
 * @param nFramesUs4OEM: number of frames each us4OEM acquires
 * @param nSequences, nFrames, nSamples, nChannels, nComponents: output shape
 */
extern "C" __global__ void arrusRemapV2(short *out, short *in, const short *fcmFrames, const char *fcmChannels,
                                      const unsigned char *fcmUs4oems, const unsigned int *frameOffsets,
                                      const unsigned int *nFramesUs4OEM, const unsigned nSequences,
                                      const unsigned nFrames, const unsigned nSamples, const unsigned nChannels,
                                      const unsigned nComponents) {
    // TODO temporarily assuming, that maximum number of components == 2
    __shared__ short tile[32][32][2]; // NOTE: this is also the runtime block size.

    // gridDim.x*32 ~= number of channels
    // gridDim.y*32 ~= number of samples
    // Input.
    int channel = blockIdx.x*32 + threadIdx.x; // logical channel
    int sample = blockIdx.y*32 + threadIdx.y;  // logical sample
    int frame = blockIdx.z; // logical frame, global in the whole batch of sequences

    int sequence = frame / nFrames;
    int localFrame = frame % nFrames;
    if (channel >= nChannels || sample >= nSamples || localFrame >= nFrames || sequence >= nSequences) {
        return;
    }
    // FCM describes here a single sequence
    int physicalChannel = fcmChannels[localFrame*nChannels + channel];
    int physicalFrame = fcmFrames[localFrame*nChannels + channel];
    int us4oem = fcmUs4oems[localFrame*nChannels + channel];
    int us4oemOffset = frameOffsets[us4oem];
    int nPhysicalFrames = nFramesUs4OEM[us4oem];

    const int nus4OEMChannels = 32;
    // physical, input
    int pSampleSize = nus4OEMChannels*nComponents;
    int pFrameSize = pSampleSize*nSamples;

    // logical, output
    int lChannelSize = nSamples*nComponents;
    int lFrameSize = nChannels*lChannelSize;
    int lSequenceSize = nFrames*lFrameSize;
    short value = 0;

    for(unsigned component = 0; component < nComponents; ++component) {
        if(physicalChannel < 0) {
            value = 0;
        }
        else {
            // Note: optimistic assumption, that we get consecutive physical channel numbers
            // (and that is satisfied e.g. for full RX aperture on esaote3 adapter, like in plane wave imaging).
            size_t indexIn = us4oemOffset*pFrameSize + sequence*nPhysicalFrames*pFrameSize
              + physicalFrame*pFrameSize + sample*pSampleSize + component*nus4OEMChannels + physicalChannel;
            value = in[indexIn];
        }
        tile[threadIdx.y][threadIdx.x][component] = value;
    }
    __syncthreads();
    // Output
    int targetSample = blockIdx.y*32 + threadIdx.x;
    int targetChannel = blockIdx.x*32 + threadIdx.y;
    for(unsigned component = 0; component < nComponents; ++component) {
        // [sequence, frame, channel, sample, component]
        size_t indexOut =
            sequence*lSequenceSize + localFrame*lFrameSize + targetChannel*lChannelSize + targetSample*nComponents + component;
        out[indexOut] = tile[threadIdx.x][threadIdx.y][component];
    }
}