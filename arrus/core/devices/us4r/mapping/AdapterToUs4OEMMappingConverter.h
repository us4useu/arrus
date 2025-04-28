#ifndef ARRUS_CORE_DEVICES_US4R_MAPPING_ADATERTOUS4OEMMAPPINGCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_MAPPING_ADATERTOUS4OEMMAPPINGCONVERTER_H

#include <utility>
#include <vector>

#include "Us4OEMApertureSplitter.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/types.h"
#include "arrus/core/devices/us4r/validators/ProbeAdapterTxRxValidator.h"

namespace arrus::devices {

class AdapterToUs4OEMMappingConverter {
public:
    /** List of sequences to upload on the OEM.*/
    using OEMSequences = std::vector<us4r::TxRxParametersSequence>;
    using SequenceByOEM = std::vector<us4r::TxRxParametersSequence>;
    using FCMByOEM = std::vector<FrameChannelMapping::RawHandle>;
    using DelayProfilesByOEM = std::unordered_map<Ordinal, std::vector<framework::NdArray>>;


    AdapterToUs4OEMMappingConverter(ProbeAdapterSettings settings, const Ordinal noems,
                                    std::vector<std::vector<uint8_t>> oemMappings, const std::optional<Ordinal> frameMetadataOEM,
                                    ChannelIdx nRxChannelsOEM)
        : settings(std::move(settings)), noems(noems), splitter{std::move(oemMappings), frameMetadataOEM, nRxChannelsOEM} {}

    std::pair<SequenceByOEM, DelayProfilesByOEM> convert(SequenceId id, const us4r::TxRxParametersSequence &seq,
                                                         const std::vector<framework::NdArray> &txDelayProfiles) {
        // Validate input sequence
        ProbeAdapterTxRxValidator validator(format("Adapter to OEMs conversion, sequence: {}", id),
                                            settings.getNumberOfChannels());
        validator.validate(seq);
        validator.throwOnErrors();

        OpId nOps = ARRUS_SAFE_CAST(seq.size(), OpId);
        ChannelIdx nChannels = settings.getNumberOfChannels();// The number of adapter channels
        sequence = seq;
        batchSize = seq.getNRepeats();
        rxApertureSize = seq.getRxApertureSize();

        // Split into multiple arrays.
        // us4oem, op number -> aperture/delays
        std::unordered_map<Ordinal, std::vector<BitMask>> txApertures, rxApertures;
        std::unordered_map<Ordinal, std::vector<std::vector<float>>> txDelaysList;
        std::unordered_map<Ordinal, std::vector<framework::NdArray>> txDelayProfilesList;
        std::unordered_map<Ordinal, std::vector<std::unordered_set<ChannelIdx>>> maskedChannelsTx;
        std::unordered_map<Ordinal, std::vector<std::unordered_set<ChannelIdx>>> maskedChannelsRx;
        // Here is an assumption, that each operation has the same size rx aperture, except RX nops.
        nFrames = seq.getNumberOfNoRxNOPs();
        // find the first non rx NOP and use it to determine rxApertureSize

        // -- Frame channel mapping stuff related to splitting each operation between available
        // modules.
        // (logical frame, logical channel) -> physical module
        // active rx channel number here refers to the LOCAL ordinal number of
        // active channel in the rx aperture; e.g. for rx aperture [0, 32, 42],
        // 0 has relative ordinal number 0, 32 has relative number 1,
        // 42 has relative number 2.
        frameModule = Eigen::MatrixXi(nFrames, rxApertureSize);
        frameModule.setConstant(FrameChannelMapping::UNAVAILABLE);
        // (logical frame, logical channel) -> actual channel on a given us4oem
        frameChannel = Eigen::MatrixXi(nFrames, rxApertureSize);
        frameChannel.setConstant(FrameChannelMapping::UNAVAILABLE);

        framework::NdArray::Shape txDelaysProfileShape = {seq.size(), Us4OEMDescriptor::N_TX_CHANNELS};

        // Initialize helper arrays.
        for (Ordinal oem = 0; oem < noems; ++oem) {
            txApertures.emplace(oem, std::vector<BitMask>(nOps));
            rxApertures.emplace(oem, std::vector<BitMask>(nOps));
            txDelaysList.emplace(oem, std::vector<std::vector<float>>(nOps));
            maskedChannelsTx.emplace(oem, std::vector<std::unordered_set<ChannelIdx>>(nOps));
            maskedChannelsRx.emplace(oem, std::vector<std::unordered_set<ChannelIdx>>(nOps));

            // Profiles.
            std::vector<framework::NdArray> txDelayProfilesForModule;
            size_t nProfiles = txDelayProfiles.size();
            for (size_t i = 0; i < nProfiles; ++i) {
                framework::NdArray emptyArray(txDelaysProfileShape, txDelayProfiles[i].getDataType(),
                                              txDelayProfiles[i].getPlacement(), txDelayProfiles[i].getName());
                txDelayProfilesForModule.push_back(std::move(emptyArray));
            }
            txDelayProfilesList.emplace(oem, txDelayProfilesForModule);
        }

        // Split Tx, Rx apertures and tx delays into sub-apertures specific for each us4oem module.
        uint32 opId = 0;
        uint32 frameNumber = 0;
        for (const auto &op : sequence) {
            const auto &txAperture = op.getTxAperture();
            const auto &rxAperture = op.getRxAperture();
            const auto &txDelays = op.getTxDelays();
            const auto &maskedAdapterChannelsTx = op.getMaskedChannelsTx();
            const auto &maskedAdapterChannelsRx = op.getMaskedChannelsRx();

            std::vector<std::vector<int32>> us4oemChannels(noems);
            std::vector<std::vector<int32>> adapterChannels(noems);

            ARRUS_REQUIRES_TRUE(txAperture.size() == rxAperture.size() && txAperture.size() == nChannels,
                                format("Tx and Rx apertures should have a size: {}", nChannels));

            for (Ordinal oem = 0; oem < noems; ++oem) {
                txApertures[oem][opId].resize(Us4OEMDescriptor::N_ADDR_CHANNELS, false);
                rxApertures[oem][opId].resize(Us4OEMDescriptor::N_ADDR_CHANNELS, false);
                txDelaysList[oem][opId].resize(Us4OEMDescriptor::N_ADDR_CHANNELS, 0.0f);
            }
            size_t adapterApCh = 0;// Adapter aperture channel
            bool isRxNop = true;

            // SPLIT tx/rx/delays between modules
            for (size_t ach = 0; ach < nChannels; ++ach) {
                // tx/rx/delays mapping stuff
                auto [oem, channel] = settings.getChannelMapping().at(ach);
                txApertures[oem][opId][channel] = txAperture[ach];
                rxApertures[oem][opId][channel] = rxAperture[ach];
                txDelaysList[oem][opId][channel] = txDelays[ach];

                // channel masking
                if(setContains(maskedAdapterChannelsTx, ARRUS_SAFE_CAST(ach, ChannelIdx))) {
                    maskedChannelsTx[oem][opId].insert(channel);
                }
                if(setContains(maskedAdapterChannelsRx, ARRUS_SAFE_CAST(ach, ChannelIdx))) {
                    maskedChannelsRx[oem][opId].insert(channel);
                }

                for (size_t i = 0; i < txDelayProfiles.size(); ++i) {
                    txDelayProfilesList[oem][i].set(opId, channel, txDelayProfiles[i].get<float>(opId, ach));
                }
                // FC Mapping stuff
                if (op.getRxAperture()[ach]) {
                    isRxNop = false;
                    frameModule(frameNumber, adapterApCh + op.getRxPadding()[0]) = oem;
                    // This will be processed further later.
                    us4oemChannels[oem].push_back(channel);
                    adapterChannels[oem].push_back(static_cast<int32>(adapterApCh + op.getRxPadding()[0]));
                    ++adapterApCh;
                }
            }
            if (!isRxNop) {
                // FCM
                // Compute rank of each us4oem RX channel (to get the "aperture" channel number).
                // The rank is needed, as the further code decomposes each op into 32-rx element ops
                // assuming, that the first 32 channels of rx aperture will be used in the first
                // op, the next 32 channels in the second op and so on.
                for (Ordinal ordinal = 0; ordinal < noems; ++ordinal) {
                    auto &uChannels = us4oemChannels[ordinal];
                    auto &aChannels = adapterChannels[ordinal];
                    auto rxApertureChannels = rank(uChannels);
                    for (size_t c = 0; c < uChannels.size(); ++c) {
                        frameChannel(frameNumber, aChannels[c]) = static_cast<int32>(rxApertureChannels[c]);
                    }
                }
                ++frameNumber;
            }
            ++opId;
        }
        SequenceByOEM fullApSeq(noems);// TX/RX sequences before splitting RX apertures.
        for (Ordinal oem = 0; oem < noems; ++oem) {
            us4r::TxRxParametersSequenceBuilder seqBuilder;
            seqBuilder.setCommon(seq);
            uint16 i = 0;
            for (const auto &op : seq) {
                us4r::TxRxParametersBuilder paramsBuilder{op};
                paramsBuilder.setTxAperture(txApertures[oem][i]);
                paramsBuilder.setRxAperture(rxApertures[oem][i]);
                paramsBuilder.setTxDelays(txDelaysList[oem][i]);
                paramsBuilder.setMaskedChannelsTx(maskedChannelsTx[oem][i]);
                paramsBuilder.setMaskedChannelsRx(maskedChannelsRx[oem][i]);
                paramsBuilder.setRxPadding(Tuple<ChannelIdx>({0, 0}));
                seqBuilder.addEntry(paramsBuilder.build());
                ++i;
            }
            fullApSeq.at(oem) = seqBuilder.build();
            // keep operations with empty tx or rx aperture - they are still a part of the larger operation
        }
        // creating 32-element subapertures
        splitResult = splitter.split(fullApSeq, txDelayProfilesList);
        return std::make_pair(splitResult->sequences, splitResult->delayProfiles);
    }

    /**
     * Returns a LOCAL (i.e. limited to the given sequence) logical to physical mapping.
     */
    const LogicalToPhysicalOp &getLogicalToPhysicalOpMap() {
        return splitResult->logicalToPhysicalMap;
    }

    /**
     * @param fcms OEM Ordinal -> FCM
     */
    FrameChannelMapping::Handle convert(const std::vector<FrameChannelMapping::Handle> &fcms) {
        uint32 currentFrameOffset = 0;
        std::vector<uint32> frameOffsets(static_cast<unsigned int>(noems), 0);
        std::vector<uint32> numberOfFrames(static_cast<unsigned int>(noems), 0);

        for (Ordinal oem = 0; oem < noems; ++oem) {
            const auto &fcm = fcms.at(oem);
            frameOffsets.at(oem) = currentFrameOffset;
            currentFrameOffset += fcm->getNumberOfLogicalFrames()*batchSize;
            numberOfFrames.at(oem) = fcm->getNumberOfLogicalFrames()*batchSize;
        }

        // generate FrameChannelMapping for the adapter output.
        FrameChannelMappingBuilder outFcBuilder(nFrames, ARRUS_SAFE_CAST(rxApertureSize, ChannelIdx));
        FrameChannelMappingBuilder::FrameNumber frameIdx = 0;
        for (const auto &op : sequence) {
            if (op.isRxNOP()) {
                continue;
            }
            uint16 activeRxChIdx = 0;
            for (auto bit : op.getRxAperture()) {
                if (bit) {
                    // Frame channel mapping determined by distributing op on multiple devices
                    auto dstModule = frameModule(frameIdx, activeRxChIdx + op.getRxPadding()[0]);
                    auto dstModuleChannel = frameChannel(frameIdx, activeRxChIdx + op.getRxPadding()[0]);

                    // if dstModuleChannel is unavailable, set channel mapping to -1 and continue
                    // unavailable dstModuleChannel means, that the given channel was virtual
                    // and has no assigned value.
                    ARRUS_REQUIRES_DATA_TYPE_E(dstModuleChannel, int8,
                                               ArrusException("Invalid dstModuleChannel data type"));
                    if (FrameChannelMapping::isChannelUnavailable((int8) dstModuleChannel)) {
                        outFcBuilder.setChannelMapping(frameIdx, activeRxChIdx + op.getRxPadding()[0], 0, 0,
                                                       FrameChannelMapping::UNAVAILABLE);
                    } else {
                        // Otherwise, we have an actual channel.
                        ARRUS_REQUIRES_TRUE_E(dstModule >= 0 && dstModuleChannel >= 0,
                                              ArrusException("Dst module and dst channel should be non-negative"));

                        // dstOp, dstChannel - frame and channel after considering that the aperture ops are
                        // into multiple smaller ops for each us4oem separately.
                        // dstOp, dstChannel - frame and channel of a given module
                        auto dstOp = splitResult->physicalFrame(dstModule, frameIdx, dstModuleChannel);
                        auto dstChannel = splitResult->physicalChannel(dstModule, frameIdx, dstModuleChannel);
                        FrameChannelMapping::Us4OEMNumber us4oem = 0;
                        FrameChannelMapping::FrameNumber dstFrame = 0;
                        int8 dstFrameChannel = -1;
                        if (!FrameChannelMapping::isChannelUnavailable(dstChannel)) {
                            auto res = fcms.at(dstModule)->getLogical(dstOp, dstChannel);
                            us4oem = arrus::devices::get<0>(res);
                            dstFrame = arrus::devices::get<1>(res);
                            dstFrameChannel = arrus::devices::get<2>(res);
                        }
                        outFcBuilder.setChannelMapping(frameIdx, activeRxChIdx + op.getRxPadding()[0], us4oem, dstFrame,
                                                       dstFrameChannel);
                    }
                    ++activeRxChIdx;
                }
            }
            ++frameIdx;
        }
        outFcBuilder.setFrameOffsets(frameOffsets);
        outFcBuilder.setNumberOfFrames(numberOfFrames);
        return outFcBuilder.build();
    }


private:
    ProbeAdapterSettings settings;
    Ordinal noems;
    Us4OEMApertureSplitter splitter;

    // Determined while converting the sequence:
    std::optional<Us4OEMApertureSplitter::Result> splitResult;
    us4r::TxRxParametersSequence sequence;
    SequenceId batchSize{0};
    OpId nFrames{0};
    ChannelIdx rxApertureSize{0};
    Eigen::MatrixXi frameModule;
    Eigen::MatrixXi frameChannel;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_MAPPING_ADATERTOUS4OEMMAPPINGCONVERTER_H
