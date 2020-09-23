#include "common.h"

#include <algorithm>
#include <execution>
#include <numeric>

namespace arrus::devices {

std::vector<TxRxParamsSequence>
splitRxAperturesIfNecessary(const std::vector<TxRxParamsSequence> &seqs) {
    // All sequences must have the same length.
    ARRUS_REQUIRES_NON_EMPTY_IAE(seqs);
    size_t seqLength = seqs.size();
    for(const auto &seq : seqs) {
        ARRUS_REQUIRES_EQUAL_IAE(seqLength, seq.size());
    }
    std::vector<TxRxParamsSequence> result;

    constexpr ChannelIdx N_RX_CHANNELS = Us4OEMImpl::N_RX_CHANNELS;
    constexpr ChannelIdx N_GROUPS =
        Us4OEMImpl::N_ADDR_CHANNELS / Us4OEMImpl::N_RX_CHANNELS;
    constexpr ChannelIdx N_ADDR_CHANNELS = Us4OEMImpl::N_ADDR_CHANNELS;

    for(const auto &seq : seqs) {
        TxRxParamsSequence resSeq;
        resSeq.reserve(seq.size());
        result.push_back(resSeq);
    }

    for(size_t i = 0; i < seqLength; ++i) {

        for(size_t seqIdx = 0; seqIdx < seqs.size(); ++seqIdx) {
            const auto &seq = seqs[seqIdx];
            const auto &op = seq[i];

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
                // Generate rx subapertures - split aperture into smaller
                // subapertures.
                std::vector<BitMask> rxSubapertures(maxSubapertureIdx);
                for(auto &subaperture : rxSubapertures) {
                    subaperture.resize(N_ADDR_CHANNELS);
                }
                for(size_t ch = 0; ch < subapertureIdxs.size(); ++ch) {

                    auto subapIdx = subapertureIdxs[ch];
                    if(subapIdx > 0) {
                        rxSubapertures[subapIdx][ch] = true;
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
                // otherwise all rx channels are empty, just pass it as is
                result[seqIdx].push_back(op);
            }
        }
        // Check if all seqs have the same size.
        // If not, pad them the NOP.
        std::vector<size_t> currentSeqSizes;
        std::transform(std::begin(result), std::end(result),
                       std::back_inserter(currentSeqSizes),
                       [](auto &v) { return v.size(); });
        size_t maxSize = *std::max_element(std::begin(currentSeqSizes),
                                           std::end(currentSeqSizes));
        for(auto& resSeq : result) {
            if(resSeq.size() < maxSize) {
                resSeq.resize(maxSize, TxRxParameters::NOP);
            }
        }
    }
    return result;
}

}