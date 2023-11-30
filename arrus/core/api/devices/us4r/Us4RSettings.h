#ifndef ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H

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

class Us4RSettings {
public:
    using ReprogrammingMode = Us4OEMSettings::ReprogrammingMode;

    /**
     * TODO deprecated, will be removed in 0.11.0
     */
    explicit Us4RSettings(std::vector<Us4OEMSettings> us4OemSettings, std::optional<HVSettings> hvSettings,
                          std::optional<Ordinal> nUs4OEMs = std::nullopt,
                          std::vector<Ordinal> adapterToUs4RModuleNumber = {},
                          int txFrequencyRange = 1,
                          std::optional<DigitalBackplaneSettings> digitalBackplaneSettings = std::nullopt
                          )
        : us4oemSettings(std::move(us4OemSettings)), hvSettings(std::move(hvSettings)),
          nUs4OEMs(nUs4OEMs), adapterToUs4RModuleNumber(std::move(adapterToUs4RModuleNumber)),
          txFrequencyRange(txFrequencyRange), digitalBackplaneSettings(std::move(digitalBackplaneSettings))
          {}

    Us4RSettings(
        ProbeAdapterSettings probeAdapterSettings,
        std::vector<ProbeSettings> probeSettings,
        RxSettings rxSettings,
        std::optional<HVSettings> hvSettings,
        std::vector<ChannelIdx> channelsMask,
        std::vector<std::vector<uint8>> us4oemChannelsMask,
        ReprogrammingMode reprogrammingMode = ReprogrammingMode::SEQUENTIAL,
        std::optional<Ordinal> nUs4OEMs = std::nullopt,
        std::vector<Ordinal> adapterToUs4RModuleNumber = {},
        bool externalTrigger = false,
        int txFrequencyRange = 1,
        std::optional<DigitalBackplaneSettings> digitalBackplaneSettings = std::nullopt
    ) : probeAdapterSettings(std::move(probeAdapterSettings)),
          probeSettings(std::move(probeSettings)),
          rxSettings(std::move(rxSettings)),
          hvSettings(std::move(hvSettings)),
          channelsMask(std::move(channelsMask)),
          us4oemChannelsMask(std::move(us4oemChannelsMask)),
          reprogrammingMode(reprogrammingMode),
          nUs4OEMs(nUs4OEMs),
          adapterToUs4RModuleNumber(std::move(adapterToUs4RModuleNumber)),
          externalTrigger(externalTrigger),
          txFrequencyRange(txFrequencyRange),
          digitalBackplaneSettings(std::move(digitalBackplaneSettings))
    {}

    Us4RSettings(
        ProbeAdapterSettings probeAdapterSettings,
        ProbeSettings probeSettings,
        RxSettings rxSettings,
        std::optional<HVSettings> hvSettings,
        std::vector<ChannelIdx> channelsMask,
        std::vector<std::vector<uint8>> us4oemChannelsMask,
        ReprogrammingMode reprogrammingMode = ReprogrammingMode::SEQUENTIAL,
        std::optional<Ordinal> nUs4OEMs = std::nullopt,
        std::vector<Ordinal> adapterToUs4RModuleNumber = {},
        bool externalTrigger = false,
        int txFrequencyRange = 1,
        std::optional<DigitalBackplaneSettings> digitalBackplaneSettings = std::nullopt
        ) : Us4RSettings(
                std::move(probeAdapterSettings),
                std::vector<ProbeSettings>{std::move(probeSettings)},
                std::move(rxSettings),
                std::move(hvSettings),
                std::move(channelsMask),
                std::move(us4oemChannelsMask),
                reprogrammingMode,
                nUs4OEMs,
                std::move(adapterToUs4RModuleNumber),
                externalTrigger,
                txFrequencyRange,
                std::move(digitalBackplaneSettings)
        )
    {}

    const std::vector<Us4OEMSettings> &getUs4OEMSettings() const {
        return us4oemSettings;
    }

    const std::optional<ProbeAdapterSettings> &
    getProbeAdapterSettings() const {
        return probeAdapterSettings;
    }

    const ProbeSettings &getProbeSettings(size_t ordinal) const {
        if(ordinal >= probeSettings.size()) {
            throw IllegalArgumentException(
                "There are no settings for probe: " + std::to_string(ordinal)
                );
        }
    }

    const std::vector<ProbeSettings> &getProbeSettingsList() const {
        return probeSettings;
    }

    /**
     * Returns probe settings for probe 0.
     * TODO (ARRUS-276) deprecated, will be removed in v0.11.0
     */
    std::optional<ProbeSettings> getProbeSettings() const {
        if(probeSettings.empty()) {
            return std::nullopt;
        }
        return getProbeSettings(0);
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

    const std::optional<Ordinal> &getNumberOfUs4oems() const {
        return nUs4OEMs;
    }

    const std::vector<Ordinal> &getAdapterToUs4RModuleNumber() const {
        return adapterToUs4RModuleNumber;
    }

    bool isExternalTrigger() const {
        return externalTrigger;
    }

    int getTxFrequencyRange() const {
        return txFrequencyRange;
    }

    const std::optional<DigitalBackplaneSettings> &getDigitalBackplaneSettings() const {
        return digitalBackplaneSettings;
    }

private:
    /* A list of settings for Us4OEMs.
     * First element configures Us4OEM:0, second: Us4OEM:1, etc. */
    std::vector<Us4OEMSettings> us4oemSettings;
    /** Probe adapter settings. Optional - when not set, at least one
     *  Us4OEMSettings must be set. When is set, the list of Us4OEM
     *  settings should be empty. */
    std::optional<ProbeAdapterSettings> probeAdapterSettings{};
    /** List of ProbeSettings to set. Optional - when is set, ProbeAdapterSettings also
     * must be available.*/
    std::vector<ProbeSettings> probeSettings;
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
     * See Us4OEMSettings::ReprogrammingMode docs for more information. */
    ReprogrammingMode reprogrammingMode;
    /** Number of us4OEMs in the us4R system. Optional, if is std::nullopt,
     * the number of us4oems is determined based on the probe adapter mapping
     * (equal to the maximum ordinal number of us4OEM). Optional, if set to
     * std::nullopt, the number of us4OEMs will be determined based on the
     * probe adapter mapping (as the maximum of us4OEM module ordinal numbers). */
    std::optional<Ordinal> nUs4OEMs = std::nullopt;
    /** The mapping from the us4OEM ordinal number in the probe adapter mapping
     * and the actual ordinal number of us4OEM. Optional, empty vector means that
     * no mapping should be applied (identity mapping). */
    std::vector<Ordinal> adapterToUs4RModuleNumber = {};
    bool externalTrigger{false};
    /** Transmit frequency range to set on us4OEM devices. Actually, TX frequency divider.
     *  Default value: 1.*/
    int txFrequencyRange = 1;
    /**
     * Digital backplane ("DBAR") settings. If not provided, DBAR will be determined based on select HV supplier.
     */
     std::optional<DigitalBackplaneSettings> digitalBackplaneSettings;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H
