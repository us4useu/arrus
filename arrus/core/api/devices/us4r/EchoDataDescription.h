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

    EchoDataDescription(FrameChannelMapping::Handle fcm, int32_t rxOffset)
    : fcm(std::move(fcm)), rxOffset(rxOffset) {}

    // TODO getters
    int32_t getRxOffset() {
        return rxOffset;
    }

    FrameChannelMapping::Handle getFrameChannelMapping() {
        return fcm;
    }

private:
    FrameChannelMapping::Handle fcm;
    int32_t rxOffset;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H
