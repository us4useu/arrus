#include "common.h"

#include <algorithm>
#include <execution>
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

std::tuple<
    std::vector<TxRxParamsSequence>,
    Eigen::Tensor<FrameChannelMapping::FrameNumber, 3>,
    Eigen::Tensor<int8, 3>
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

    // us4oem ordinal number -> current frame idx
    std::vector<FrameNumber> currentFrameIdx(seqs.size(), 0);
    for(size_t opIdx = 0; opIdx < seqLength; ++opIdx) {
        for(size_t us4oemIdx = 0; us4oemIdx < seqs.size(); ++us4oemIdx) {
            const auto &seq = seqs[us4oemIdx];
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
                        // -1 because subapIdx starts from one
                        opDestOp(us4oemIdx, opIdx, opActiveChannel) = FrameNumber(currentFrameIdx[us4oemIdx] + subapIdx - 1);
                        ARRUS_REQUIRES_TRUE_E(
                            opActiveChannel <= (std::numeric_limits<int8>::max)(),
                            arrus::ArrusException(
                                "Number of active rx elements should not exceed 32."));
                        opDestChannel(us4oemIdx, opIdx, opActiveChannel) =
                            static_cast<int8>(subopActiveChannels[subapIdx-1]);
                        ++opActiveChannel;
                        ++subopActiveChannels[subapIdx-1];
                    }
                }
                // generate ops from subapertures
                for(auto &subaperture : rxSubapertures) {
                    result[us4oemIdx].emplace_back(
                        op.getTxAperture(), op.getTxDelays(), op.getTxPulse(),
                        subaperture, // Modified
                        op.getRxSampleRange(), op.getRxDecimationFactor(), op.getPri(),
                        op.getRxPadding());
                }
            } else {
                // we have a single rx aperture, or all rx channels are empty,
                // just pass the operator as is
                // NOTE: we push_back even if the op is rx nop
                result[us4oemIdx].push_back(op);
                // FC mapping
                ChannelIdx opActiveChannel = 0;
                for(auto bit : op.getRxAperture()) {
                    if(bit) {
                        opDestOp(us4oemIdx, opIdx, opActiveChannel) =
                            currentFrameIdx[us4oemIdx];
                        ARRUS_REQUIRES_TRUE_E(
                            opActiveChannel <= (std::numeric_limits<int8>::max)(),
                            arrus::ArrusException(
                                "Number of active rx elements should not exceed 32."));
                        opDestChannel(us4oemIdx, opIdx, opActiveChannel)
                            = static_cast<int8>(opActiveChannel);
                        ++opActiveChannel;
                    }
                }
            }
            if(us4oemIdx == 0) {
                // NOTE: for us4OEM:0, even if it is RX nop, the results of this
                // rx NOP will be transferred from us4OEM to host memory,
                // to get the frame metadata.
                maxSubapertureIdx = max((ChannelIdx)1, maxSubapertureIdx);
            }
            currentFrameIdx[us4oemIdx] += maxSubapertureIdx;
        }
        // Check if all seqs have the same size.
        // If not, pad them with a rx NOP.
        std::vector<size_t> currentSeqSizes;
        std::transform(std::begin(result), std::end(result),
                       std::back_inserter(currentSeqSizes),
                       [](auto &v) { return v.size(); });
        size_t maxSize = *std::max_element(std::begin(currentSeqSizes),
                                           std::end(currentSeqSizes));

        for(auto& resSeq : result) {
            if(resSeq.size() < maxSize) {
                // create rxnop copy from the last element of this sequence
                // note, that even if the last element is rx nop it should be added
                // in this method in some of the code above.
                resSeq.resize(maxSize, TxRxParameters::createRxNOPCopy(resSeq[resSeq.size()-1]));
            }
        }
    }
    return std::make_tuple(result, opDestOp, opDestChannel);
}

}