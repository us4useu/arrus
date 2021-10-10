#ifndef ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H

#include <utility>
#include <map>
#include <ostream>

#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/HVSettings.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::devices {

class Us4RSettings {
public:
    using ReprogrammingMode = Us4OEMSettings::ReprogrammingMode;

    explicit Us4RSettings(std::vector<Us4OEMSettings> us4OemSettings, std::optional<HVSettings> hvSettings)
        : us4oemSettings(std::move(us4OemSettings)), hvSettings(std::move(hvSettings)) {}

    Us4RSettings(
        ProbeAdapterSettings probeAdapterSettings,
        ProbeSettings probeSettings,
        RxSettings rxSettings,
        std::optional<HVSettings> hvSettings,
        std::vector<ChannelIdx> channelsMask,
        std::vector<std::vector<uint8>> us4oemChannelsMask,
        ReprogrammingMode reprogrammingMode = ReprogrammingMode::SEQUENTIAL)
        : probeAdapterSettings(std::move(probeAdapterSettings)),
          probeSettings(std::move(probeSettings)),
          rxSettings(std::move(rxSettings)),
          hvSettings(std::move(hvSettings)),
          channelsMask(std::move(channelsMask)),
          us4oemChannelsMask(std::move(us4oemChannelsMask)),
          reprogrammingMode(reprogrammingMode) {}

    const std::vector<Us4OEMSettings> &getUs4OEMSettings() const {
        return us4oemSettings;
    }

    const std::optional<ProbeAdapterSettings> &
    getProbeAdapterSettings() const {
        return probeAdapterSettings;
    }

    const std::optional<ProbeSettings> &getProbeSettings() const {
        return probeSettings;
    }

    const std::optional<RxSettings> &getRxSettings() const {
        return rxSettings;
    }

    const std::optional<HVSettings> &getHVSettings() const {
        return hvSettings;
    }

    const std::vector<ChannelIdx> &getChannelsMask() const {
        return channelsMask;
    }

    const std::vector<std::vector<uint8>> &getUs4OEMChannelsMask() const {
        return us4oemChannelsMask;
    }

    ReprogrammingMode getReprogrammingMode() const {
        return reprogrammingMode;
    }

private:
    /* A list of settings for Us4OEMs.
     * First element configures Us4OEM:0, second: Us4OEM:1, etc. */
    std::vector<Us4OEMSettings> us4oemSettings;
    /** Probe adapter settings. Optional - when not set, at least one
     *  Us4OEMSettings must be set. When is set, the list of Us4OEM
     *  settings should be empty. */
    std::optional<ProbeAdapterSettings> probeAdapterSettings{};
    /** ProbeSettings to set. Optional - when is set, ProbeAdapterSettings also
     * must be available.*/
    std::optional<ProbeSettings> probeSettings{};
    /** Required when no Us4OEM settings are set. */
    std::optional<RxSettings> rxSettings;
    /** Optional (us4r devices may have externally controlled hv suppliers. */
    std::optional<HVSettings> hvSettings;
    /** A list of channels that should be turned off in the us4r system.
     * Note that the **channel numbers start from 0**.*/
    std::vector<ChannelIdx> channelsMask;
    /** A list of channels masks to apply on given us4oems.
     * Currently us4oem channels are used for double check only.
     * The administrator has to provide us4oem channels masks that confirms to
     * the system us4r channels, and this way we reduce the chance of mistake. */
    std::vector<std::vector<uint8>> us4oemChannelsMask;
    /** Reprogramming mode applied to all us4OEMs.
     * See Us4OEMSettings::ReprogrammingMode docs for more information.*/
    ReprogrammingMode reprogrammingMode;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H
