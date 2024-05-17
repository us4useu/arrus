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
#include "arrus/core/api/devices/us4r/Bitstream.h"

namespace arrus::devices {

class Us4RSettings {
public:
    using ReprogrammingMode = Us4OEMSettings::ReprogrammingMode;

    Us4RSettings(
        ProbeAdapterSettings probeAdapterSettings,
        std::vector<ProbeSettings> probeSettings,
        RxSettings rxSettings,
        std::optional<HVSettings> hvSettings,
        std::vector<std::vector<ChannelIdx>> channelsMask,
        ReprogrammingMode reprogrammingMode = ReprogrammingMode::SEQUENTIAL,
        std::optional<Ordinal> nUs4OEMs = std::nullopt,
        std::vector<Ordinal> adapterToUs4RModuleNumber = {},
        bool externalTrigger = false,
        int txFrequencyRange = 1,
        std::optional<DigitalBackplaneSettings> digitalBackplaneSettings = std::nullopt,
        std::vector<Bitstream> bitstreams = std::vector<Bitstream>()
    ) : probeAdapterSettings(std::move(probeAdapterSettings)),
          probeSettings(std::move(probeSettings)),
          rxSettings(std::move(rxSettings)),
          hvSettings(std::move(hvSettings)),
          channelsMask(std::move(channelsMask)),
          reprogrammingMode(reprogrammingMode),
          nUs4OEMs(nUs4OEMs),
          adapterToUs4RModuleNumber(std::move(adapterToUs4RModuleNumber)),
          externalTrigger(externalTrigger),
          txFrequencyRange(txFrequencyRange),
          digitalBackplaneSettings(std::move(digitalBackplaneSettings)),
          bitstreams(std::move(bitstreams))
    {}

    Us4RSettings(
        ProbeAdapterSettings probeAdapterSettings,
        ProbeSettings probeSettings,
        RxSettings rxSettings,
        std::optional<HVSettings> hvSettings,
        std::vector<ChannelIdx> probe0ChannelsMask,
        ReprogrammingMode reprogrammingMode = ReprogrammingMode::SEQUENTIAL,
        std::optional<Ordinal> nUs4OEMs = std::nullopt,
        std::vector<Ordinal> adapterToUs4RModuleNumber = {},
        bool externalTrigger = false,
        int txFrequencyRange = 1,
        std::optional<DigitalBackplaneSettings> digitalBackplaneSettings = std::nullopt,
        std::vector<Bitstream> bitstreams = std::vector<Bitstream>()
        ) : Us4RSettings(
                std::move(probeAdapterSettings),
                std::vector<ProbeSettings>{std::move(probeSettings)},
                std::move(rxSettings),
                std::move(hvSettings),
                {std::move(probe0ChannelsMask)},
                reprogrammingMode,
                nUs4OEMs,
                std::move(adapterToUs4RModuleNumber),
                externalTrigger,
                txFrequencyRange,
                std::move(digitalBackplaneSettings),
                std::move(bitstreams)
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
        return probeSettings.at(ordinal);
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

    Ordinal getNumberOfProbes() const {
        return (Ordinal)probeSettings.size();
    }

    const std::optional<RxSettings> &getRxSettings() const {
        return rxSettings;
    }

    const std::optional<HVSettings> &getHVSettings() const {
        return hvSettings;
    }

    /**
     * Returns channels mask to be applied for Probe:0 TX/RX apertures.
     * DEPRECATED (v0.11.0): please use getChannelsMask(probeNr).
     */
    const std::vector<ChannelIdx> &getChannelsMask() const {
        return getChannelsMaskForProbe(0);
    }

    const std::vector<ChannelIdx> &getChannelsMaskForProbe(Ordinal probeNr) const {
        return channelsMask.at(probeNr);
    }

    const std::vector<std::vector<ChannelIdx>> &getChannelsMaskForAllProbes() const {
        return channelsMask;
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

    const std::vector<Bitstream> &getBitstreams() const { return bitstreams; }

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
     * This is list of lists; each list represents what channels of the
     * ultrasound interface (probe) should be turned off.
     * channelsMask[i] is a channels mask for the i-th probe (Probe:i).
     * Note that the **channel numbers start from 0**.*/
    std::vector<std::vector<ChannelIdx>> channelsMask;
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
     /**
      * Bitstream definitions.
      */
     std::vector<Bitstream> bitstreams;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H
