#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMRXMAPPINGBUILDER_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMRXMAPPINGBUILDER_H

namespace arrus::devices {

struct RxMappingSetting {
    std::unordered_map<uint16, uint16> mapping;
    std::vector<std::bitset<Us4OEMImpl::N_ADDR_CHANNELS>> apertures;
    FrameChannelMapping::Handle fcm;
    uint16 nRxMappings;
};

class Us4OEMRxMappingBuilder {

    void add(const TxRxParametersSequence &sequence) {}

    RxMappingSetting build() {
        std::unordered_map<uint16, uint16> firingToRxMappingId;
        std::unordered_map<std::vector<uint8>, uint16, ContainerHash<std::vector<uint8>>> rxMappings;
        // FC mapping
        auto numberOfOutputFrames = seq.getNumberOfNoRxNOPs();
        if (acceptRxNops) {
            // We transfer all module frames due to possible metadata stored in the frame (if enabled).
            numberOfOutputFrames = ARRUS_SAFE_CAST(seq.size(), ChannelIdx);
        }
        FrameChannelMappingBuilder fcmBuilder(numberOfOutputFrames, N_RX_CHANNELS);
        // Rx apertures after taking into account possible conflicts in Rx channel mapping.
        std::vector<Us4OEMBitMask> outputRxApertures;

        uint16 rxMapId = rxMapIdOffset;
        uint16 opId = 0;
        uint16 noRxNopId = 0;

        for (const auto &op : seq.getParameters()) {
            // Considering rx nops: rx channel mapping will be equal [0, 1,.. 31].
            // Index of rx aperture channel (0, 1...32) -> us4oem physical channel
            // nullopt means that given channel is missing (conflicting with some other channel or is masked)
            std::vector<std::optional<uint8>> mapping;
            std::unordered_set<uint8> channelsUsed;
            // Convert rx aperture + channel mapping -> new rx aperture (with conflicting channels turned off).
            std::bitset<N_ADDR_CHANNELS> outputRxAperture;
            // Us4OEM channel number: values from 0-127
            uint8 channel = 0;
            // Number of Us4OEM active channel, values from 0-31
            uint8 onChannel = 0;
            bool isRxNop = true;
            for (const auto isOn : op.getRxAperture()) {
                if (isOn) {
                    isRxNop = false;
                    ARRUS_REQUIRES_TRUE(onChannel < N_RX_CHANNELS, "Up to 32 active rx channels can be set.");
                    // Physical channel number, values 0-31
                    auto rxChannel = channelMapping[channel];
                    rxChannel = rxChannel % N_RX_CHANNELS;
                    if (!setContains(channelsUsed, rxChannel) && !setContains(this->channelsMask, channel)) {
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
                    fcmBuilder.setChannelMapping(frameNumber, onChannel,
                                                 FrameChannelMapping::Us4OEMNumber(getDeviceId().getOrdinal()),
                                                 frameNumber, (int8) (mapping.size() - 1));
                    ++onChannel;
                }
                ++channel;
            }
            outputRxApertures.push_back(outputRxAperture);

            // GENERATE RX MAPPING.
            std::vector<uint8> rxMapping;
            // - Determine unused channels.
            std::list<uint8> unusedChannels;
            for (uint8 i = 0; i < N_RX_CHANNELS; ++i) {
                if (!setContains(channelsUsed, i)) {
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
            while (rxMapping.size() != 32) {
                rxMapping.push_back(unusedChannels.front());
                unusedChannels.pop_front();
            }

            // SET RX MAPPING.
            auto mappingIt = rxMappings.find(rxMapping);
            if (mappingIt == std::end(rxMappings)) {
                // - If this is a brand-new mapping -- create it on us4OEM.
                rxMappings.emplace(rxMapping, rxMapId);
                firingToRxMappingId.emplace(opId, rxMapId);
                // Set channel mapping
                ARRUS_REQUIRES_TRUE(rxMapping.size() == N_RX_CHANNELS,
                                    format("Invalid size of the RX channel mapping to set: {}", rxMapping.size()));
                ARRUS_REQUIRES_TRUE(
                    rxMapId < 128,
                    format("128 different rx mappings can be loaded only, deviceId: {}.", getDeviceId().toString()));
                ius4oem->SetRxChannelMapping(rxMapping, rxMapId);
                ++rxMapId;
            } else {
                // - Otherwise use the existing one.
                firingToRxMappingId.emplace(opId, mappingIt->second);
            }
            ++opId;
            if (!isRxNop) {
                ++noRxNopId;
            }
        }
    }
};



}

#endif//ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMRXMAPPINGBUILDER_H
