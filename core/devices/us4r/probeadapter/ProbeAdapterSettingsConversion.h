#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERSETTINGSCONVERSION_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERSETTINGSCONVERSION_H

#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"

namespace arrus {

class ProbeAdapterSettingsConversion final {

    static std::vector<Us4OEMSettings> toUs4OEMSettings(
            const ProbeAdapterSettings probeAdapterSettings) {
        //convert
    }

private:
    ProbeAdapterSettingsConversion() = default;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERSETTINGSCONVERSION_H
