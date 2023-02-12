#ifndef ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H
#define ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H

#include <utility>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"

namespace arrus::devices {

class EchoDataDescription {
public:
    using Handle = std::shared_ptr<EchoDataDescription>;

    EchoDataDescription(FrameChannelMapping::Handle fcm, uint32_t rxOffset)
    : fcm(std::move(fcm)), rxOffset(rxOffset) {}

    FrameChannelMapping::Handle fcm;
    uint32_t rxOffset;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H
