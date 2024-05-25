#ifndef ARRUS_CORE_DEVICES_US4R_MAPPING_US4OEMAPERTURESPLITTER_H
#define ARRUS_CORE_DEVICES_US4R_MAPPING_US4OEMAPERTURESPLITTER_H

#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>

#include "arrus/core/common/aperture.h"
#include "arrus/core/common/collections.h"

#include "arrus/core/external/eigen/Tensor.h"

namespace arrus::devices {

class Us4OEMApertureSplitter {
public:
    using SequenceByOEM = std::vector<us4r::TxRxParametersSequence>;
    using SequenceBuilderByOEM = std::vector<us4r::TxRxParametersSequenceBuilder>;
    using DelayProfileByOEM = std::unordered_map<Ordinal, std::vector<framework::NdArray>>;

    struct Result {
        // recaculated sequences
        SequenceByOEM sequences;
        // a mapping (module, input op index, rx channel) -> output frame number
        Eigen::Tensor<FrameChannelMapping::FrameNumber, 3> physicalFrame;
        // a mapping (module, input op index, rx channel) -> output frame rx channel
        Eigen::Tensor<int8, 3> physicalChannel;
        std::unordered_map<Ordinal, std::vector<framework::NdArray>> delayProfiles;
    };

    Us4OEMApertureSplitter(std::vector<std::vector<uint8_t>> oemMappings, const std::optional<Ordinal> frameMetadataOEM,
                           const ChannelIdx nRxChannels)
        : oemMappings(std::move(oemMappings)), frameMetadataOEM(frameMetadataOEM), nRxChannels(nRxChannels) {}

    /**
    * Splits each tx/rx operation into multiple ops so that each rx aperture
    * does not include the same rx channel multiple times.
    *
    * This function is intended to be used for Us4OEM TxRxs only!
    *
    * Note: us4oems have 32 rx channels, however 128 rx channels are addressable;
    * each addressable rx channel 'i' is connected to us4oem channel 'i modulo 32',
    * so for example us4oem channel 0 can handle the output addressable channels
    * 0, 32, 64, 96; only one of these channels can be set in a single Rx aperture.
    *
    * `Seqs` input parameter is a vector of sequences that will be loaded on
    * us4oem:0, us4oem:1, etc. This function outputs updated sequences so that
    * there are no conflicting rx channels. All of the output sequences have the
    * same length - e.g. if seqs[0] first tx/rx operation must be split into
    * 4 tx/rx ops, and seqs[1] first op must be split into 2 tx/rx ops only,
    * the second sequence will extended by NOP TxRxParameters.
    *
    * @param sequences tx/rx sequences to recalculate
    */
    Result split(const SequenceByOEM &sequences, const DelayProfileByOEM &delayProfiles) const {
        using FrameNumber = FrameChannelMapping::FrameNumber;
        // All sequences must have the same length.
        ARRUS_REQUIRES_NON_EMPTY_IAE(sequences);
        size_t seqLength = sequences[0].size();
        for (const auto &seq : sequences) {
            ARRUS_REQUIRES_EQUAL_IAE(seqLength, seq.size());
        }
        Ordinal noems = ARRUS_SAFE_CAST(sequences.size(), Ordinal);
        SequenceBuilderByOEM sequenceBuilders(noems);
        // Initialize builders.
        for (Ordinal oem = 0; oem < noems; ++oem) {
            sequenceBuilders.at(oem).setCommon(sequences.at(oem));
        }

        // Find the maximum rx aperture size
        ChannelIdx maxRxApertureSize = getMaximumRxAperture(sequences);
        FrameChannelMapping::FrameNumber numberOfFrames = getNumberOfFrames(sequences);
        // (module, logical frame, logical rx channel) -> physical frame
        Eigen::Tensor<FrameChannelMapping::FrameNumber, 3> opDestOp(sequences.size(), numberOfFrames,
                                                                    maxRxApertureSize);
        // (module, logical frame, logical rx channel) -> physical rx channel
        Eigen::Tensor<int8, 3> opDestChannel(sequences.size(), numberOfFrames, maxRxApertureSize);
        std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> outputTxDelayProfiles;
        std::vector<size_t> srcOpIdx;// srcOpIdx[output op idx] = input op idx (before splitting into sub-apertures)

        opDestOp.setZero();
        opDestChannel.setConstant(FrameChannelMapping::UNAVAILABLE);

        ChannelIdx nGroups = Us4OEMDescriptor::N_ADDR_CHANNELS / nRxChannels;
        constexpr ChannelIdx N_ADDR_CHANNELS = Us4OEMDescriptor::N_ADDR_CHANNELS;

        std::vector<std::vector<uint8_t>> us4oemP2LMappings(oemMappings.size());
        int i = 0;
        for (const auto &mapping : oemMappings) {
            us4oemP2LMappings[i++] = revertMapping<uint8_t>(mapping);
        }

        // us4oem ordinal number -> current frame idx
        std::vector<FrameNumber> currentFrameIdx(sequences.size(), 0);
        size_t frameIdx = 0;
        // For each operation
        for (size_t opIdx = 0; opIdx < seqLength; ++opIdx) {// For each TX/RX
            // Determine if the op is Rx NOP.
            bool isRxNOP = true;
            for(size_t oem = 0; oem < noems; ++oem) {
                const auto &op = sequences.at(oem).at(opIdx);
                if(! op.isRxNOP()) {
                    isRxNOP = false;
                    break;
                }
            }
            for (size_t oem = 0; oem < noems; ++oem) {      // for each OEM
                const auto &seq = sequences.at(oem);
                const auto &op = seq.at(opIdx);

                // Split rx aperture, if necessary.
                // subaperture number starts from 1, 0 means that the channel
                // should be inactive.
                std::vector<ChannelIdx> subapertureIdxs(op.getRxAperture().size());
                for (ChannelIdx ch = 0; ch < nRxChannels; ++ch) {
                    ChannelIdx subaperture = 1;
                    for (ChannelIdx group = 0; group < nGroups; ++group) {
                        // Us4OEM Physical address
                        ChannelIdx physicalIdx = group * nRxChannels + ch;
                        // Us4OEM Logical address
                        ChannelIdx logicalIdx = us4oemP2LMappings[oem][physicalIdx];
                        if (op.getRxAperture()[logicalIdx]) {
                            // channel active
                            subapertureIdxs[logicalIdx] = subaperture++;
                        } else {
                            // channel inactive
                            subapertureIdxs[logicalIdx] = 0;
                        }
                    }
                }
                ChannelIdx maxSubapertureIdx =
                    *std::max_element(std::begin(subapertureIdxs), std::end(subapertureIdxs));
                if (maxSubapertureIdx > 1) {
                    // Split aperture into smaller subapertures (Muxing).
                    std::vector<BitMask> rxSubapertures(maxSubapertureIdx);
                    for (auto &subaperture : rxSubapertures) {
                        subaperture.resize(N_ADDR_CHANNELS);
                    }
                    std::vector<std::unordered_set<ChannelIdx>> maskedChannelsRx(maxSubapertureIdx);

                    long long opActiveChannel = 0;
                    std::vector<ChannelIdx> subopActiveChannels(maxSubapertureIdx, 0);
                    for (size_t ch = 0; ch < subapertureIdxs.size(); ++ch) {
                        auto subapIdx = subapertureIdxs[ch];
                        if (subapIdx > 0) {
                            rxSubapertures[subapIdx - 1][ch] = true;
                            // Channel masking
                            if(setContains(op.getMaskedChannelsRx(),ARRUS_SAFE_CAST(ch, ChannelIdx))) {
                                maskedChannelsRx.at(subapIdx-1).insert(ch);
                            }
                            // FC mapping
                            // -1 because subapIdx starts from one
                            opDestOp(oem, frameIdx, opActiveChannel) = FrameNumber(currentFrameIdx[oem] + subapIdx - 1);
                            ARRUS_REQUIRES_TRUE_E(
                                opActiveChannel <= (std::numeric_limits<int8>::max)(),
                                arrus::ArrusException("Number of active rx elements should not exceed 32."));
                            opDestChannel(oem, frameIdx, opActiveChannel) =
                                static_cast<int8>(subopActiveChannels[subapIdx - 1]);
                            ++opActiveChannel;
                            ++subopActiveChannels[subapIdx - 1];
                        }
                    }
                    // generate ops from subapertures
                    for(size_t subap = 0; subap < rxSubapertures.size(); ++subap) {
                        auto &subaperture = rxSubapertures.at(subap);
                        auto rxMask = maskedChannelsRx.at(subap);
                        us4r::TxRxParametersBuilder builder(op);
                        builder.setRxAperture(subaperture);
                        builder.setMaskedChannelsRx(rxMask);
                        sequenceBuilders.at(oem).addEntry(builder.build());
                    }
                } else {
                    // we have a single rx aperture, or all rx channels are empty,
                    // just pass the operator as is
                    // NOTE: we push_back even if the op is rx nop
                    sequenceBuilders.at(oem).addEntry(op);
                    // FC mapping
                    ChannelIdx opActiveChannel = 0;
                    for (auto bit : op.getRxAperture()) {
                        if (bit) {
                            opDestOp(oem, frameIdx, opActiveChannel) = currentFrameIdx[oem];
                            ARRUS_REQUIRES_TRUE_E(
                                opActiveChannel <= (std::numeric_limits<int8>::max)(),
                                arrus::ArrusException("Number of active rx elements should not exceed 32."));
                            opDestChannel(oem, frameIdx, opActiveChannel) = static_cast<int8>(opActiveChannel);
                            ++opActiveChannel;
                        }
                    }
                }
                currentFrameIdx[oem] += maxSubapertureIdx;
            }
            // Check if all seqs have the same size.
            // If not, pad them with a rx NOP.
            std::vector<size_t> currentSizes;
            std::transform(std::begin(sequenceBuilders), std::end(sequenceBuilders), std::back_inserter(currentSizes),
                           [](const auto &builder) { return builder.getCurrent().size(); });
            size_t maxSize = *std::max_element(std::begin(currentSizes), std::end(currentSizes));
            for (auto &b : sequenceBuilders) {
                if (b.getCurrent().size() < maxSize) {
                    // create rxnop copy from the last element of this sequence
                    // note, that even if the last element is rx nop it should be added
                    // in this method in some of the code above.
                    b.resize(maxSize, us4r::TxRxParameters::createRxNOPCopy(b.getCurrent().getLastOp()));
                }
            }
            // NOTE: for us4OEM that acquires metadata, even if it is RX nop, the results of this
            // rx NOP will be transferred from us4OEM to host memory,
            // to get the frame metadata. Therefore, we need to increase
            // the number of frames a given element contains.
            if(frameMetadataOEM.has_value()) {
                currentFrameIdx[frameMetadataOEM.value()] = FrameNumber(maxSize);
            }
            srcOpIdx.resize(maxSize, opIdx);
            if(!isRxNOP) {
                frameIdx++;
            }
        }

        // Map to target TX delays (after splitting to sub-apertures).
        // Optimization: if we simply have 1-1 mapping between input and output sequences, just return the input txDelays.
        SequenceByOEM result;
        for(auto &b: sequenceBuilders) {
            result.emplace_back(b.build());
        }
        if (areConsecutive(srcOpIdx) || delayProfiles.empty()) {
            return Result{result, opDestOp, opDestChannel, delayProfiles};
        } else {
            for (size_t seqIdx = 0; seqIdx < result.size(); ++seqIdx) {
                size_t nOps = result[seqIdx].size();
                framework::NdArray::Shape shape{nOps, Us4OEMDescriptor::N_TX_CHANNELS};
                framework::NdArray::DataType dataType = framework::NdArray::DataType::FLOAT32;
                std::vector<::arrus::framework::NdArray> outputProfiles;
                for (auto &profile : delayProfiles.at(static_cast<uint16_t>(seqIdx))) {
                    const DeviceId &placement = profile.getPlacement();
                    const std::string &name = profile.getName();
                    ::arrus::framework::NdArray outputProfile(shape, dataType, placement, name);
                    for (size_t opIdx = 0; opIdx < nOps; ++opIdx) {
                        for (size_t ch = 0; ch < shape.get(1); ++ch) {
                            outputProfile.set(opIdx, ch, profile.get<float>(srcOpIdx[opIdx], ch));
                        }
                    }
                    outputProfiles.push_back(outputProfile);
                }
                outputTxDelayProfiles.emplace(static_cast<uint16_t>(seqIdx), outputProfiles);
            }
            return Result{result, opDestOp, opDestChannel, delayProfiles};
        }
    }

private:
    static ChannelIdx getMaximumRxAperture(const std::vector<us4r::TxRxParametersSequence> &seqs) {
        ChannelIdx maxElementSize = 0;
        for (const auto &seq : seqs) {
            for (const auto &op : seq) {
                ChannelIdx n = getNumberOfActiveChannels(op.getRxAperture());
                if (n > maxElementSize) {
                    maxElementSize = n;
                }
            }
        }
        return maxElementSize;
    }

    static FrameChannelMapping::FrameNumber getNumberOfFrames(const std::vector<us4r::TxRxParametersSequence> &seqs) {
        FrameChannelMapping::FrameNumber numberOfFrames = 0;
        auto numberOfOps = seqs[0].size();
        for (size_t opIdx = 0; opIdx < numberOfOps; ++opIdx) {
            for (const auto &seq : seqs) {
                if (!seq.at(opIdx).isRxNOP()) {
                    ++numberOfFrames;
                    break;
                }
            }
        }
        return numberOfFrames;
    }

    template<typename T> std::vector<T> revertMapping(const std::vector<T> &mapping) const {
        std::vector<T> result(mapping.size());
        for (size_t i = 0; i < mapping.size(); ++i) {
            result[mapping[i]] = (uint8_t) i;
        }
        return result;
    }

    // us4oem ordinal -> us4oem logical to physical mapping
    std::vector<std::vector<uint8_t>> oemMappings;
    std::optional<Ordinal> frameMetadataOEM{std::nullopt};
    ChannelIdx nRxChannels;
};

}// namespace arrus::devices

#endif// ARRUS_CORE_DEVICES_US4R_MAPPING_US4OEMAPERTURESPLITTER_H
