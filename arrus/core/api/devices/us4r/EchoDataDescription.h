#ifndef ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H
#define ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H

#include <utility>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"

namespace arrus::devices {

class EchoDataDescription {
public:
    using Handle = std::unique_ptr<EchoDataDescription>;
    using SharedHandle = std::shared_ptr<EchoDataDescription>;

    EchoDataDescription(FrameChannelMapping::SharedHandle fcm, int32_t rxOffset)
    : fcm_(std::move(fcm)), rxOffset_(rxOffset) {}

    // TODO getters
    int32_t getRxOffset() {
        return rxOffset_;
    }

    FrameChannelMapping::SharedHandle getFrameChannelMapping() {
        return fcm_;
    }

    void setFrameChannelMapping(FrameChannelMapping::SharedHandle fcm) { 
        fcm_ = std::move(fcm); 
    }

private:
    FrameChannelMapping::SharedHandle fcm_;
    int32_t rxOffset_;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H
