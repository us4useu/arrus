#include "common.h"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>

#include "arrus/core/common/aperture.h"
#include "arrus/core/common/collections.h"

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
    Eigen::Tensor<int8, 3>,
    std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>>
>
splitRxAperturesIfNecessary(const std::vector<TxRxParamsSequence> &seqs,
                            const std::vector<std::vector<uint8_t>> &us4oemL2PMappings,
                            const std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> &inputTxDelayProfiles,
                            Ordinal frameMetadataOem = 0) {
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
    std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> outputTxDelayProfiles;
    std::vector<size_t> srcOpIdx; // srcOpIdx[output op idx] = input op idx (before splitting into sub-apertures)

    opDestOp.setZero();
    opDestChannel.setConstant(FrameChannelMapping::UNAVAILABLE);

    constexpr ChannelIdx N_RX_CHANNELS = Us4OEMImpl::N_RX_CHANNELS;
    constexpr ChannelIdx N_GROUPS = Us4OEMImpl::N_ADDR_CHANNELS / Us4OEMImpl::N_RX_CHANNELS;
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
    size_t frameIdx = 0;
    for(size_t opIdx = 0; opIdx < seqLength; ++opIdx) { // For each TX/RX
	// Determine, if this is RX NOP.
	bool isRxNOP = true;
	for (size_t seqIdx = 0; seqIdx < seqs.size(); ++seqIdx)	{
	    if(!seqs[seqIdx][opIdx].isRxNOP()) {	
		// Not an RX NOP.
		isRxNOP = false;
		break;
	    }
	}
        for(size_t seqIdx = 0; seqIdx < seqs.size(); ++seqIdx) { // for each OEM
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
                // Split aperture into smaller subapertures (Muxing).
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
                        opDestOp(seqIdx, frameIdx, opActiveChannel) = FrameNumber(currentFrameIdx[seqIdx] + subapIdx - 1);
                        ARRUS_REQUIRES_TRUE_E(
                            opActiveChannel <= (std::numeric_limits<int8>::max)(),
                            arrus::ArrusException(
                                "Number of active rx elements should not exceed 32."));
                        opDestChannel(seqIdx, frameIdx, opActiveChannel) =
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
                        opDestOp(seqIdx, frameIdx, opActiveChannel) = currentFrameIdx[seqIdx];
                        ARRUS_REQUIRES_TRUE_E(
                            opActiveChannel <= (std::numeric_limits<int8>::max)(),
                            arrus::ArrusException(
                                "Number of active rx elements should not exceed 32."));
                        opDestChannel(seqIdx, frameIdx, opActiveChannel)
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
        std::transform(std::begin(result), std::end(result), std::back_inserter(currentSeqSizes),
                       [](auto &v) { return v.size(); });
        size_t maxSize = *std::max_element(std::begin(currentSeqSizes), std::end(currentSeqSizes));
        for(auto& resSeq : result) {
            if(resSeq.size() < maxSize) {
                // create rxnop copy from the last element of this sequence
                // note, that even if the last element is rx nop it should be added
                // in this method in some of the code above.
                resSeq.resize(maxSize, TxRxParameters::createRxNOPCopy(resSeq[resSeq.size()-1]));
            }
        }
        // NOTE: for us4OEM that acquires metadata, even if it is RX nop, the results of this
        // rx NOP will be transferred from us4OEM to host memory,
        // to get the frame metadata. Therefore we need to increase
        // the number of frames a given element contains.
        currentFrameIdx[frameMetadataOem] = FrameNumber(maxSize);

        srcOpIdx.resize(maxSize, opIdx);
	if(!isRxNOP) {
           frameIdx++;
	}
    }

    // Map to target TX delays (after splitting to sub-apertures).
    // Optimization: if we simply have 1-1 mapping between input and output sequences, just return the input txDelays.
    if(::arrus::areConsecutive(srcOpIdx) || inputTxDelayProfiles.empty()) {
        return std::make_tuple(result, opDestOp, opDestChannel, inputTxDelayProfiles);
    }
    else {
        for(size_t seqIdx = 0; seqIdx < result.size(); ++seqIdx) {
            size_t nOps = result[seqIdx].size();
            ::arrus::framework::NdArray::Shape shape{nOps, Us4OEMImpl::N_TX_CHANNELS};
            ::arrus::framework::NdArray::DataType dataType = framework::NdArray::DataType::FLOAT32;
            std::vector<::arrus::framework::NdArray> outputProfiles;
            for(auto &profile: inputTxDelayProfiles.at(static_cast<uint16_t>(seqIdx))) {
                const DeviceId &placement = profile.getPlacement();
                const std::string &name = profile.getName();
                ::arrus::framework::NdArray outputProfile(shape, dataType, placement, name);
                for(size_t opIdx = 0; opIdx < nOps; ++opIdx) {
                    for(size_t ch = 0; ch < shape.get(1); ++ch) {
                        outputProfile.set(opIdx, ch, profile.get<float>(srcOpIdx[opIdx], ch));
                    }
                }
                outputProfiles.push_back(outputProfile);
            }
            outputTxDelayProfiles.emplace(static_cast<uint16_t>(seqIdx), outputProfiles);
        }
        return std::make_tuple(result, opDestOp, opDestChannel, outputTxDelayProfiles);
    }
}

}

