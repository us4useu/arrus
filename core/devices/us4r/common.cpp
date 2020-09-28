#include "common.h"

#include <algorithm>
#include <execution>
#include <numeric>
#include <tuple>

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

std::tuple<
    std::vector<TxRxParamsSequence>,
    Eigen::Tensor<int32, 3>,
    Eigen::Tensor<int32, 3>
>
splitRxAperturesIfNecessary(const std::vector<TxRxParamsSequence> &seqs) {
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
    Eigen::Tensor<int32, 3> opDestOp(seqs.size(), numberOfFrames, maxRxApertureSize);
    // (module, logical frame, logical rx channel) -> physical rx channel
    Eigen::Tensor<int32, 3> opDestChannel(seqs.size(), numberOfFrames, maxRxApertureSize);
    opDestOp.setConstant(FrameChannelMapping::UNAVAILABLE);
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

    // us4oem ordinal number -> current frame idx
    std::vector<FrameNumber> currentFrameIdx(seqs.size(), 0);
    for(size_t opIdx = 0; opIdx < seqLength; ++opIdx) {
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
                    ChannelIdx addrIdx = group * N_RX_CHANNELS + ch;
                    if(op.getRxAperture()[addrIdx]) {
                        // channel active
                        subapertureIdxs[addrIdx] = subaperture++;
                    } else {
                        // channel inactive
                        subapertureIdxs[addrIdx] = 0;
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
                        // -1 because subapIdx starts from zero
                        opDestOp(seqIdx, opIdx, opActiveChannel) = FrameNumber(currentFrameIdx[seqIdx] + subapIdx - 1);
                        opDestChannel(seqIdx, opIdx, opActiveChannel) = subopActiveChannels[subapIdx-1];
                        ++opActiveChannel;
                        ++subopActiveChannels[subapIdx-1];
                    }
                }
                // generate ops from subapertures
                for(auto &subaperture : rxSubapertures) {
                    result[seqIdx].emplace_back(
                        op.getTxAperture(), op.getTxDelays(), op.getTxPulse(),
                        subaperture, // Modified
                        op.getRxSampleRange(), op.getRxDecimationFactor(), op.getPri());
                }
            } else {
                // we have a single rx aperture, or all rx channels are empty,
                // just pass the operator as is
                result[seqIdx].push_back(op);
                // FC mapping
                ChannelIdx opActiveChannel = 0;
                for(auto bit : op.getRxAperture()) {
                    if(bit) {
                        opDestOp(seqIdx, opIdx, opActiveChannel) = currentFrameIdx[seqIdx];
                        opDestChannel(seqIdx, opIdx, opActiveChannel) = opActiveChannel;
                        ++opActiveChannel;
                    }
                }
            }
            currentFrameIdx[seqIdx] += maxSubapertureIdx;
        }
        // Check if all seqs have the same size.
        // If not, pad them with a NOP.
        std::vector<size_t> currentSeqSizes;
        std::transform(std::begin(result), std::end(result),
                       std::back_inserter(currentSeqSizes),
                       [](auto &v) { return v.size(); });
        size_t maxSize = *std::max_element(std::begin(currentSeqSizes),
                                           std::end(currentSeqSizes));
        for(auto& resSeq : result) {
            if(resSeq.size() < maxSize) {
                resSeq.resize(maxSize, TxRxParameters::US4OEM_NOP);
            }
        }
    }
    return std::make_tuple(result, opDestOp, opDestChannel);
}

}