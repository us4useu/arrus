#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMRXMAPPING_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMRXMAPPING_H

#include "Us4OEMDescriptor.h"
#include "Us4OEMImplBase.h"
#include "arrus/common/utils.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

#include <list>

namespace arrus::devices {

class Us4OEMRxMappingRegisterBuilder;

class Us4OEMRxMappingRegister {
public:
    using RxMapId = uint16;
    using RxMap = std::vector<uint8>;
    using RxAperture = std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS>;

    // Disable copy constructor (fcms is not copiable)
    Us4OEMRxMappingRegister(const Us4OEMRxMappingRegister &other) = delete;
    Us4OEMRxMappingRegister &operator=(const Us4OEMRxMappingRegister &other) = delete;

    Us4OEMRxMappingRegister(Us4OEMRxMappingRegister &&other) noexcept
        : mappings(std::move(other.mappings)), opToRxMappingId(std::move(other.opToRxMappingId)),
          rxApertures(std::move(other.rxApertures)), fcms(std::move(other.fcms)) {}

    Us4OEMRxMappingRegister &operator=(Us4OEMRxMappingRegister &&other) noexcept {
        if (this == &other)
            return *this;
        mappings = std::move(other.mappings);
        opToRxMappingId = std::move(other.opToRxMappingId);
        rxApertures = std::move(other.rxApertures);
        fcms = std::move(other.fcms);
        return *this;
    }

    const std::unordered_map<RxMapId, RxMap> &getMappings() const { return mappings; }

    RxMapId getMapId(SequenceId sequenceId, OpId opId) const { return opToRxMappingId.at(std::make_pair(sequenceId, opId)); }

    RxMap getMap(SequenceId sequenceId, OpId opId) {
        auto mapId = getMapId(sequenceId, opId);
        return getMappings().at(mapId);
    }

    RxAperture getRxAperture(SequenceId sequenceId, OpId opId) const {
        return rxApertures.at(std::make_pair(sequenceId, opId));
    }
    std::vector<FrameChannelMapping::Handle> acquireFCMs() { return std::move(fcms); }

private:
    friend class Us4OEMRxMappingRegisterBuilder;

    Us4OEMRxMappingRegister() = default;

    void insert(SequenceId sequenceId, OpId opId, RxAperture aperture) {
        rxApertures.emplace(std::make_pair(sequenceId, opId), aperture);
    }
    void insert(SequenceId sequenceId, OpId opId, RxMapId rxMapId) {
        opToRxMappingId.emplace(std::make_pair(sequenceId, opId), rxMapId);
    }

    void insert(RxMapId id, RxMap rxMap) { mappings.emplace(id, std::move(rxMap)); }

    void push_back(FrameChannelMapping::Handle fcm) { fcms.push_back(std::move(fcm)); }

    std::unordered_map<RxMapId, RxMap> mappings;
    std::unordered_map<std::pair<SequenceId, OpId>, RxMapId, PairHash<SequenceId, OpId>> opToRxMappingId;
    // Rx apertures after taking into account possible conflicts in Rx channel mapping.
    std::unordered_map<std::pair<SequenceId, OpId>, RxAperture, PairHash<SequenceId, OpId>> rxApertures;
    std::vector<FrameChannelMapping::Handle> fcms;
};

class Us4OEMRxMappingRegisterBuilder {
public:
    using RxAperture = Us4OEMRxMappingRegister::RxAperture;
    using RxMapId = Us4OEMRxMappingRegister::RxMapId;
    using RxMap = Us4OEMRxMappingRegister::RxMap;

    Us4OEMRxMappingRegisterBuilder(FrameChannelMapping::Us4OEMNumber oem, bool acceptRxNops,
                                   const std::vector<uint8_t> &channelMapping,
                                   ChannelIdx nRxChannels)
        : oem(oem), acceptRxNops(acceptRxNops), channelMapping(channelMapping), nRxChannels(nRxChannels) {}

    void add(const std::vector<us4r::TxRxParametersSequence> &sequences) {
        for (size_t sequenceId = 0; sequenceId < sequences.size(); ++sequenceId) {
            add(ARRUS_SAFE_CAST(sequenceId, SequenceId), sequences.at(sequenceId));
        }
    }

    void add(SequenceId sequenceId, const us4r::TxRxParametersSequence &sequence) {
        auto numberOfOutputFrames = sequence.getNumberOfNoRxNOPs();
        if (acceptRxNops) {
            // We transfer all module frames due to possible metadata stored in the frame (if enabled).
            numberOfOutputFrames = ARRUS_SAFE_CAST(sequence.size(), ChannelIdx);
        }
        FrameChannelMappingBuilder fcmBuilder(numberOfOutputFrames, nRxChannels);
        OpId opId = 0;
        OpId noRxNopId = 0;

        for (const auto &op : sequence.getParameters()) {
            // Considering rx nops: rx channel mapping will be equal [0, 1,.. 31].
            // Index of rx aperture channel (0, 1...32) -> us4oem physical channel
            // nullopt means that given channel is missing (conflicting with some other channel or is masked)
            std::vector<std::optional<uint8>> mapping;
            std::unordered_set<uint8> channelsUsed;
            // Convert rx aperture + channel mapping -> new rx aperture (with conflicting channels turned off).
            RxAperture outputRxAperture;
            // Us4OEM channel number: values from 0-127
            uint8 channel = 0;
            // Number of Us4OEM active channel, values from 0-31
            uint8 onChannel = 0;
            bool isRxNop = true;
            for (const auto isOn : op.getRxAperture()) {
                if (isOn) {
                    isRxNop = false;
                    ARRUS_REQUIRES_TRUE(onChannel < nRxChannels,
                                        format("Up to {} active rx channels can be set.", nRxChannels));
                    // Physical channel number, values 0-31
                    auto rxChannel = channelMapping[channel];
                    rxChannel = rxChannel % nRxChannels;
                    if (!setContains(channelsUsed, rxChannel) && !setContains(op.getMaskedChannelsRx(), static_cast<ChannelIdx>(channel))) {
                        // This channel is OK.
                        // STRATEGY: if there are conflicting/masked rx channels, keep the
                        // first one (with the lowest channel number), turn off all
                        // the rest. Turn off conflicting channels.
                        outputRxAperture[channel] = true;
                        mapping.emplace_back(rxChannel);
                        channelsUsed.insert(rxChannel);
                    } else {
                        // This channel is not OK.
                        mapping.emplace_back(std::nullopt);
                    }
                    auto frameNumber = acceptRxNops ? opId : noRxNopId;
                    fcmBuilder.setChannelMapping(frameNumber, onChannel, oem, frameNumber, (int8) (mapping.size() - 1));
                    ++onChannel;
                }
                ++channel;
            }
            // Register aperture.
            result.insert(sequenceId, opId, outputRxAperture);

            // GENERATE RX MAPPING.
            std::vector<uint8> rxMapping = buildMapping(mapping, channelsUsed);

            // SET RX MAPPING.
            auto mappingIt = rxMappings.find(rxMapping);
            if (mappingIt == std::end(rxMappings)) {
                ARRUS_REQUIRES_TRUE(rxMapping.size() == nRxChannels,
                                    format("Invalid size of the RX channel mapping: {}", rxMapping.size()));
                ARRUS_REQUIRES_TRUE(currentMapId < 128,
                                    format("128 different rx mappings can be loaded only, oem: {}.", oem));
                // - This is a brand-new mapping -- create it on us4OEM.
                rxMappings.emplace(rxMapping, currentMapId);
                result.insert(currentMapId, rxMapping);
                result.insert(sequenceId, opId, currentMapId);
                ++currentMapId;
            } else {
                // - Otherwise use the existing one.
                result.insert(sequenceId, opId, mappingIt->second);
            }
            ++opId;
            if (!isRxNop) {
                ++noRxNopId;
            }
        }
        result.push_back(fcmBuilder.build());
    }

    Us4OEMRxMappingRegister build() { return std::move(result); }

private:
    std::vector<uint8> buildMapping(std::vector<std::optional<uint8>> mapping,
                                    std::unordered_set<uint8> channelsInUse) {
        std::vector<uint8> rxMapping;
        // - Determine unused channels.
        std::list<uint8> unusedChannels;
        for (uint8 i = 0; i < nRxChannels; ++i) {
            if (!setContains(channelsInUse, i)) {
                unusedChannels.push_back(i);
            }
        }
        for (auto &dstChannel : mapping) {
            if (!dstChannel.has_value()) {
                rxMapping.push_back(unusedChannels.front());
                unusedChannels.pop_front();
            } else {
                rxMapping.push_back(dstChannel.value());
            }
        }
        // - Move all the non-active channels to the end of mapping.
        while (rxMapping.size() != nRxChannels) {
            rxMapping.push_back(unusedChannels.front());
            unusedChannels.pop_front();
        }
        return rxMapping;
    }

    FrameChannelMapping::Us4OEMNumber oem;
    bool acceptRxNops;
    std::vector<uint8_t> channelMapping;
    // Genearted.
    std::unordered_map<std::vector<uint8>, uint16, ContainerHash<std::vector<uint8>>> rxMappings;
    RxMapId currentMapId{0};
    Us4OEMRxMappingRegister result;
    ChannelIdx nRxChannels;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMRXMAPPING_H
