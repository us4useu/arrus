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
    using fcm = std::shared_ptr<FrameChannelMapping>;
    using rxOffset = uint32;

private:

};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_ECHODATADESCRIPTION_H
