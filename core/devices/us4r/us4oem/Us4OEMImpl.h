#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H

#include <utility>
#include <iostream>

#include "arrus/common/format.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"

namespace arrus {

class Us4OEMImpl : public Us4OEM {
public:
    // TGC constants.
    static constexpr float TGC_ATTENUATION_RANGE = 40.0f;
    static constexpr float TGC_SAMPLING_FREQUENCY = 1e6;
    static constexpr size_t TGC_N_SAMPLES = 1022;

    // Number of tx/rx channels.
    static constexpr ChannelIdx N_TX_CHANNELS = 128;
    static constexpr ChannelIdx N_RX_CHANNELS = 32;
    static constexpr ChannelIdx ACTIVE_CHANNEL_GROUP_SIZE = 8;
    static constexpr ChannelIdx N_ACTIVE_CHANNEL_GROUPS =
        N_TX_CHANNELS / ACTIVE_CHANNEL_GROUP_SIZE;


    Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem, BitMask activeChannelGroups,
               std::vector<uint8_t> channelMapping);

    ~Us4OEMImpl() override;

    void setTxRxSequence();


private:
    Logger::Handle logger;
    IUs4OEMHandle ius4oem;
    BitMask activeChannelGroups;
    std::vector<uint8_t> channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
