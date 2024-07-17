#ifndef ARRUS_CORE_API_DEVICES_US4R_BITSTREAM_H
#define ARRUS_CORE_API_DEVICES_US4R_BITSTREAM_H

#include <utility>
#include <map>
#include <ostream>

#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "RxSettings.h"
#include "arrus/core/api/devices/us4r/HVSettings.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/us4r/DigitalBackplaneSettings.h"

namespace arrus::devices {

class Bitstream {

public:
    Bitstream(const std::vector<uint8> &levels, const std::vector<uint16> &periods)
        : levels(levels), periods(periods) {}

    const std::vector<uint8> &getLevels() const { return levels; }
    const std::vector<uint16> &getPeriods() const { return periods; }

private:
    std::vector<uint8> levels;
    std::vector<uint16> periods;
};
}

#endif//ARRUS_CORE_API_DEVICES_US4R_BITSTREAM_H
