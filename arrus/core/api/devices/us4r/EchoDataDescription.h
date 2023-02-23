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
        : fcm(std::move(fcm)), rxOffset(rxOffset) {}

    int32_t getRxOffset() const { return rxOffset; }

    FrameChannelMapping::SharedHandle getFrameChannelMapping() { return fcm; }
private:
    FrameChannelMapping::SharedHandle fcm;
    int32_t rxOffset;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H
