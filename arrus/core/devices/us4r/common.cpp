#include "common.h"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>

#include "arrus/core/common/aperture.h"

namespace arrus::devices {

static ChannelIdx getMaximumRxAperture(const std::vector<TxRxParamsSequence> &seqs) {
    ChannelIdx maxElementSize = 0;
    for(const auto& seq: seqs) {
        for(const auto &op : seq) {
            ChannelIdx n = getNumberOfActiveChannels(op.getRxAperture());
            if(n > maxElementSize) {
                maxElementSize = n;
            }
        }
    }
    return maxElementSize;
}

static FrameChannelMapping::FrameNumber
getNumberOfFrames(const std::vector<TxRxParamsSequence> &seqs) {
    FrameChannelMapping::FrameNumber numberOfFrames = 0;
    auto numberOfOps = seqs[0].size();
    for(size_t opIdx = 0; opIdx < numberOfOps; ++opIdx) {
        for(const auto & seq : seqs) {
            if(!seq[opIdx].isRxNOP()) {
                ++numberOfFrames;
                break;
            }
        }
    }
    return numberOfFrames;
}

template<typename T>
std::vector<T> revertMapping(const std::vector<T> &mapping) {
    std::vector<T> result(mapping.size());
    for(size_t i = 0; i < mapping.size(); ++i) {
        result[mapping[i]] = (uint8_t) i;
    }
    return result;
}

/**
 *
 * @param seqs  us4oem ordinal -> tx/rx sequence
 * @param us4oemL2PMappings us4oem ordinal -> us4oem logical to physical mapping
 * @return
 */
std::tuple<
    std::vector<TxRxParamsSequence>,
    Eigen::Tensor<FrameChannelMapping::FrameNumber, 3>,
    Eigen::Tensor<int8, 3>
>
splitRxAperturesIfNecessary(const std::vector<TxRxParamsSequence> &seqs,
                            const std::vector<std::vector<uint8_t>> &us4oemL2PMappings) {
    using FrameNumber = FrameChannelMapping::FrameNumber;
    // All sequences must have the same length.
    ARRUS_REQUIRES_NON_EMPTY_IAE(seqs);
    size_t seqLength = seqs[0].size();
    for(const auto &seq : seqs) {
        ARRUS_REQUIRES_EQUAL_IAE(seqLength, seq.size());
    }
    std::vector<TxRxParamsSequence> result;

    // Find the maximum rx aperture size
    ChannelIdx maxRxApertureSize = getMaximumRxAperture(seqs);
    FrameChannelMapping::FrameNumber numberOfFrames = getNumberOfFrames(seqs);
    // (module, logical frame, logical rx channel) -> physical frame
    Eigen::Tensor<FrameChannelMapping::FrameNumber, 3> opDestOp(seqs.size(), numberOfFrames, maxRxApertureSize);
    // (module, logical frame, logical rx channel) -> physical rx channel
    Eigen::Tensor<int8, 3> opDestChannel(seqs.size(), numberOfFrames, maxRxApertureSize);
    opDestOp.setZero();
    opDestChannel.setConstant(FrameChannelMapping::UNAVAILABLE);

    constexpr ChannelIdx N_RX_CHANNELS = Us4OEMImpl::N_RX_CHANNELS;
    constexpr ChannelIdx N_GROUPS =
        Us4OEMImpl::N_ADDR_CHANNELS / Us4OEMImpl::N_RX_CHANNELS;
    constexpr ChannelIdx N_ADDR_CHANNELS = Us4OEMImpl::N_ADDR_CHANNELS;

    for(const auto &seq : seqs) {
        TxRxParamsSequence resSeq;
        resSeq.reserve(seq.size());
        result.push_back(resSeq);
    }

    std::vector<std::vector<uint8_t>> us4oemP2LMappings(us4oemL2PMappings.size());
    int i = 0;
    for(auto &mapping: us4oemL2PMappings) {
        us4oemP2LMappings[i++] = revertMapping<uint8_t>(mapping);
    }

    // us4oem ordinal number -> current frame idx
    std::vector<FrameNumber> currentFrameIdx(seqs.size(), 0);
    // For each operation
    for(size_t opIdx = 0; opIdx < seqLength; ++opIdx) {
        // For each module
        for(size_t seqIdx = 0; seqIdx < seqs.size(); ++seqIdx) {
            const auto &seq = seqs[seqIdx];
            const auto &op = seq[opIdx];

            // Split rx aperture, if necessary.
            // subaperture number starts from 1, 0 means that the channel
            // should be inactive.
            std::vector<ChannelIdx> subapertureIdxs(op.getRxAperture().size());
            for(ChannelIdx ch = 0; ch < N_RX_CHANNELS; ++ch) {
                ChannelIdx subaperture = 1;
                for(ChannelIdx group = 0; group < N_GROUPS; ++group) {
                    // Us4OEM Physical address
                    ChannelIdx physicalIdx = group*N_RX_CHANNELS + ch;
                    // Us4OEM Logical address
                    ChannelIdx logicalIdx = us4oemP2LMappings[seqIdx][physicalIdx];
                    if(op.getRxAperture()[logicalIdx]) {
                        // channel active
                        subapertureIdxs[logicalIdx] = subaperture++;
                    } else {
                        // channel inactive
                        subapertureIdxs[logicalIdx] = 0;
                    }
                }
            }
            ChannelIdx maxSubapertureIdx = *std::max_element(
                std::begin(subapertureIdxs), std::end(subapertureIdxs));
            if(maxSubapertureIdx > 1) {
                // Split aperture into smaller subapertures.
                std::vector<BitMask> rxSubapertures(maxSubapertureIdx);
                for(auto &subaperture : rxSubapertures) {
                    subaperture.resize(N_ADDR_CHANNELS);
                }

                long long opActiveChannel = 0;
                std::vector<ChannelIdx> subopActiveChannels(maxSubapertureIdx, 0);
                for(size_t ch = 0; ch < subapertureIdxs.size(); ++ch) {
                    auto subapIdx = subapertureIdxs[ch];
                    if(subapIdx > 0) {
                        rxSubapertures[subapIdx-1][ch] = true;
                        // FC mapping
                        // -1 because subapIdx starts from one
                        opDestOp(seqIdx, opIdx, opActiveChannel) = FrameNumber(currentFrameIdx[seqIdx] + subapIdx - 1);
                        ARRUS_REQUIRES_TRUE_E(
                            opActiveChannel <= (std::numeric_limits<int8>::max)(),
                            arrus::ArrusException(
                                "Number of active rx elements should not exceed 32."));
                        opDestChannel(seqIdx, opIdx, opActiveChannel) =
                            static_cast<int8>(subopActiveChannels[subapIdx-1]);
                        ++opActiveChannel;
                        ++subopActiveChannels[subapIdx-1];
                    }
                }
                // generate ops from subapertures
                for(auto &subaperture : rxSubapertures) {
                    result[seqIdx].emplace_back(
                        op.getTxAperture(), op.getTxDelays(), op.getTxPulse(),
                        subaperture, // Modified
                        op.getRxSampleRange(), op.getRxDecimationFactor(), op.getPri(),
                        op.getRxPadding());
                }
            } else {
                // we have a single rx aperture, or all rx channels are empty,
                // just pass the operator as is
                // NOTE: we push_back even if the op is rx nop
                result[seqIdx].push_back(op);
                // FC mapping
                ChannelIdx opActiveChannel = 0;
                for(auto bit : op.getRxAperture()) {
                    if(bit) {
                        opDestOp(seqIdx, opIdx, opActiveChannel) =
                            currentFrameIdx[seqIdx];
                        ARRUS_REQUIRES_TRUE_E(
                            opActiveChannel <= (std::numeric_limits<int8>::max)(),
                            arrus::ArrusException(
                                "Number of active rx elements should not exceed 32."));
                        opDestChannel(seqIdx, opIdx, opActiveChannel)
                            = static_cast<int8>(opActiveChannel);
                        ++opActiveChannel;
                    }
                }
            }
            currentFrameIdx[seqIdx] += maxSubapertureIdx;
        }
        // Check if all seqs have the same size.
        // If not, pad them with a rx NOP.
        std::vector<size_t> currentSeqSizes;
        std::transform(std::begin(result), std::end(result),
                       std::back_inserter(currentSeqSizes),
                       [](auto &v) { return v.size(); });
        size_t maxSize = *std::max_element(std::begin(currentSeqSizes),
                                           std::end(currentSeqSizes));
        int us4oemOrdinal = 0;
        for(auto& resSeq : result) {
            if(resSeq.size() < maxSize) {
                // create rxnop copy from the last element of this sequence
                // note, that even if the last element is rx nop it should be added
                // in this method in some of the code above.
                if(us4oemOrdinal == 0) {
                    // NOTE: for us4OEM:0, even if it is RX nop, the results of this
                    // rx NOP will be transferred from us4OEM to host memory,
                    // to get the frame metadata. Therefore we need to increase
                    // the number of frames a given element contains.
                    currentFrameIdx[seqIdx] += (maxSize-resSeq.size());
                }
                resSeq.resize(maxSize, TxRxParameters::createRxNOPCopy(resSeq[resSeq.size()-1]));

            }
            us4oemOrdinal++;
        }
    }
    return std::make_tuple(result, opDestOp, opDestChannel);
}

}
