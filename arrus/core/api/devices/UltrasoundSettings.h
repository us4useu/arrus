#ifndef ARRUS_ARRUS_CORE_API_DEVICES_US4R_ULTRASOUNDSETTINGS_H
#define ARRUS_ARRUS_CORE_API_DEVICES_US4R_ULTRASOUNDSETTINGS_H

#include <vector>
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/Us4RSettings.h"

namespace arrus::devices {
/**
 * General settings for ultrasound device.
 * The device configured as ultrasound
 */
class UltrasoundSettings {
public:

private:
    RxSettings rxSettings;
    std::vector<ChannelIdx> channelsMask;
    // backend settings: one of:
    Us4RSettings us4RSettings;
};
}

#endif//ARRUS_ARRUS_CORE_API_DEVICES_US4R_ULTRASOUNDSETTINGS_H
