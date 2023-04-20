#ifndef ARRUS_CORE_DEVICES_US4R_US4RSETTINGSCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_US4RSETTINGSCONVERTER_H


#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"

namespace arrus::devices {

class Us4RSettingsConverter {
public:
    /**
     * Converts given Us4R settings to us4oem specific settings and
     * appropriately remapped probe adapter settings.
     *
     * Use the returned settings when us4oems are connected in the us4r
     * to given adapter and probe.
     *
     * Channels mask - a list of PROBE channels that has to be masked.
     * Channel numbers starts from 0!
     * This function converts the probe channels mask to us4oem channels masks.
     */
    virtual
    std::pair<std::vector<Us4OEMSettings>, ProbeAdapterSettings>
    convertToUs4OEMSettings(const ProbeAdapterSettings &probeAdapterSettings,
                            const ProbeSettings &probeSettings,
                            const RxSettings &rxSettings,
                            const std::vector<ChannelIdx> &channelsMask,
                            Us4OEMSettings::ReprogrammingMode reprogrammingMode,
                            std::optional<Ordinal> nUs4OEMs,
                            const std::vector<Ordinal> &adapterToUs4RModuleNr,
                            int txFrequencyRange) = 0;

    virtual ~Us4RSettingsConverter() = default;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RSETTINGSCONVERTER_H
