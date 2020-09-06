#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H

#include <utility>
#include <iostream>

#include "arrus/core/common/logging.h"
#include "arrus/common/format.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"

namespace arrus {

class Us4OEMImpl : public Us4OEM {
    public:
    Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
               BitMask activeChannelGroups,
               std::vector<uint8_t> channelMapping)
        : Us4OEM(id), ius4oem(std::move(ius4oem)),
          activeChannelGroups(std::move(activeChannelGroups)),
          channelMapping(std::move(channelMapping)) {}

    static constexpr float TGC_RANGE = 40.0f;
    static constexpr ChannelIdx N_TX_CHANNELS = 128;
    static constexpr ChannelIdx N_RX_CHANNELS = 32;
    static constexpr ChannelIdx ACTIVE_CHANNEL_GROUP_SIZE = 8;
    static constexpr ChannelIdx N_ACTIVE_CHANNEL_GROUPS =
        N_TX_CHANNELS / ACTIVE_CHANNEL_GROUP_SIZE;
    static constexpr size_t N_TGC_SAMPLES = 1022;

    ~Us4OEMImpl() override {
        try {
            // TODO replace default logger with device specific logger here
            // still, remember to handle exceptions
            getDefaultLogger()->log(LogSeverity::DEBUG,
                                    arrus::format("Destroying {} instance",
                                                  getDeviceId().toString()));
        } catch(std::exception &e) {
            std::cerr << arrus::format("Exception while calling destructor: {}",
                                       e.what()) << std::endl;
        }
    }

    void startTrigger() override {

    }

    void stopTrigger() override {

    }

    private:
    IUs4OEMHandle ius4oem;
    BitMask activeChannelGroups;
    std::vector<uint8_t> channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
