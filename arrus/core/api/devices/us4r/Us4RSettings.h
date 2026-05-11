#ifndef ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H

#include <utility>
#include <map>
#include <ostream>

#include "arrus/core/api/devices/us4r/Us4OEMInterrupt.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "RxSettings.h"
#include "Us4RTxRxLimits.h"
#include "arrus/core/api/devices/us4r/HVSettings.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/us4r/DigitalBackplaneSettings.h"
#include "arrus/core/api/devices/us4r/Bitstream.h"
#include "arrus/core/api/devices/us4r/WatchdogSettings.h"
#include "arrus/core/api/devices/us4r/HVPSFuseSettings.h"

namespace arrus::devices {

/**
 * Us4R device settings.
 *
 * @param probeAdapterSettings Probe adapter settings. Optional - when not set, at least one
 * Us4OEMSettings must be set. When is set, the list of Us4OEM
 * settings should be empty.
 * @param probeSettings List of ProbeSettings to set. Optional - when is set, ProbeAdapterSettings also
 * @param rxSettings initial RX (AFE) settings
 * @param hvSettings high-voltage supplier settings, Optional (us4r devices may have externally controlled hv suppliers).
 * @param channelsMask A set of channels that should be turned off in the us4r system. This is list of lists; each list represents what channels of the ultrasound interface (probe) should be turned off. channelsMask[i] is a channels mask for the i-th probe (Probe:i). Note that the **channel numbers start from 0
 * @param reprogrammingMode reprogramming mode applied to all us4OEMs. See Us4OEMSettings::ReprogrammingMode docs for more information.
 * @param nUs4OEMs number of us4OEMs in the us4R system. Optional, if is std::nullopt, the number of us4oems is determined based on the probe adapter mapping (equal to the maximum ordinal number of us4OEM). Optional, if set to std::nullopt, the number of us4OEMs will be determined based on the probe adapter mapping (as the maximum of us4OEM module ordinal numbers).
 * @param adapterToUs4RModuleNumber The mapping from the us4OEM ordinal number in the probe adapter mapping and the actual ordinal number of us4OEM. Optional, empty vector means that no mapping should be applied (identity mapping).
 * @param externalTrigger whether the external trigger (TRIG INPUT) should be enabled
 * @param txFrequencyRange Transmit frequency range to set on us4OEM devices. Actually, TX frequency divider.
 * @param digitalBackplaneSettings digital backplane ("DBAR") settings. If not provided, the software will try to determine DBAR model based on select HV supplier.
 * @param bitstreams us4OEM I/O bitstream definitions
 * @param limits TX/RX constraints to apply on the system (e.g. minimum/maximum voltage, etc.).
 * @param watchdogSettings us4OEM+ watchdog settings.
 * @param allowDuplicateOEMIds whether we should allow to run system with duplicate OEM ids (e.g. due to connectivity issues).
 * @param hvpsFuseSettings HVPS fuse settings to use; nullopt means that the default settings should be used
 */
class Us4RSettings {
public:
    using ReprogrammingMode = Us4OEMSettings::ReprogrammingMode;

    Us4RSettings(
        ProbeAdapterSettings probeAdapterSettings,
        std::vector<ProbeSettings> probeSettings,
        RxSettings rxSettings,
        std::optional<HVSettings> hvSettings,
        std::vector<std::unordered_set<ChannelIdx>> channelsMask,
        ReprogrammingMode reprogrammingMode = ReprogrammingMode::SEQUENTIAL,
        std::optional<Ordinal> nUs4OEMs = std::nullopt,
        std::vector<Ordinal> adapterToUs4RModuleNumber = {},
        bool externalTrigger = false,
        int txFrequencyRange = 1,
        std::optional<DigitalBackplaneSettings> digitalBackplaneSettings = std::nullopt,
        std::vector<Bitstream> bitstreams = std::vector<Bitstream>(),
        std::optional<Us4RTxRxLimits> limits = std::nullopt,
        WatchdogSettings watchdogSettings = WatchdogSettings::defaultSettings(),
        bool allowDuplicateOEMIds = true,
        bool maskDVDDInterrupt = false,
        std::optional<HVPSFuseSettings> hvpsFuseSettings = std::nullopt,
        Us4OEMInterruptCallbacksMap interruptCallbacks = {}
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
          bitstreams(std::move(bitstreams)),
          limits(std::move(limits)),
          watchdogSettings(std::move(watchdogSettings)),
          allowDuplicateOEMIds(allowDuplicateOEMIds),
          maskDVDDInterrupt(maskDVDDInterrupt),
          hvpsFuseSettings(std::move(hvpsFuseSettings)),
          interruptCallbacks(std::move(interruptCallbacks))
    {}

    Us4RSettings(
        ProbeAdapterSettings probeAdapterSettings,
        ProbeSettings probeSettings,
        RxSettings rxSettings,
        std::optional<HVSettings> hvSettings,
        std::unordered_set<ChannelIdx> probe0ChannelsMask,
        ReprogrammingMode reprogrammingMode = ReprogrammingMode::SEQUENTIAL,
        std::optional<Ordinal> nUs4OEMs = std::nullopt,
        std::vector<Ordinal> adapterToUs4RModuleNumber = {},
        bool externalTrigger = false,
        int txFrequencyRange = 1,
        std::optional<DigitalBackplaneSettings> digitalBackplaneSettings = std::nullopt,
        std::vector<Bitstream> bitstreams = std::vector<Bitstream>(),
        std::optional<Us4RTxRxLimits> limits = std::nullopt,
        WatchdogSettings watchdogSettings = WatchdogSettings::defaultSettings(),
        bool allowDuplicateOEMIds = true,
        bool maskDVDDInterrupt = false,
        std::optional<HVPSFuseSettings> hvpsFuseSettings = std::nullopt,
        Us4OEMInterruptCallbacksMap interruptCallbacks = {}
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
                std::move(bitstreams),
                std::move(limits),
                std::move(watchdogSettings),
                allowDuplicateOEMIds,
                maskDVDDInterrupt,
                std::move(hvpsFuseSettings),
                std::move(interruptCallbacks)
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
     * TODO (ARRUS-276) deprecated, will be removed in v0.12.0
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
    const std::unordered_set<ChannelIdx> &getChannelsMask() const {
        return getChannelsMaskForProbe(0);
    }

    const std::unordered_set<ChannelIdx> &getChannelsMaskForProbe(Ordinal probeNr) const {
        return channelsMask.at(probeNr);
    }

    const std::vector<std::unordered_set<ChannelIdx>> &getChannelsMaskForAllProbes() const {
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

    const std::optional<Us4RTxRxLimits> &getTxRxLimits() const { return limits; }

    const WatchdogSettings &getWatchdogSettings() const { return watchdogSettings; }

    bool isAllowDuplicateOEMIds() const { return allowDuplicateOEMIds; }

    bool isDVDDInterruptMasked() const { return maskDVDDInterrupt; }

    const std::optional<HVPSFuseSettings> &getHVPSFuseSettings() const { return hvpsFuseSettings; }

    const Us4OEMInterruptCallbacksMap &getInterruptCallbacks() const { return interruptCallbacks; }

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
    /** A set of channels that should be turned off in the us4r system.
     * This is list of lists; each list represents what channels of the
     * ultrasound interface (probe) should be turned off.
     * channelsMask[i] is a channels mask for the i-th probe (Probe:i).
     * Note that the **channel numbers start from 0**.*/
    std::vector<std::unordered_set<ChannelIdx>> channelsMask;
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
     /**
      * TxRx limits to apply for in this session with us4R. Optional, by default the us4us-defined limits are applied.
      */
     std::optional<Us4RTxRxLimits> limits{std::nullopt};
     /** OEM watchdog settings */
     WatchdogSettings watchdogSettings{1.0f, 1.1f, 1.0f};
     /**
      * False value means that an error will raised in case the OEM id is non-unique. True value means that only
      * a warning message will be logged.
      * This parameter should be set to false in case your system has non-unique ID, but still you would like to run
      * the software.
      */
     bool allowDuplicateOEMIds{true};

     /**
     * False value means that Pulser DVDD interrput is enabled and error will raised when it occurs
     */
     bool maskDVDDInterrupt{false};

     /**
      * HVPS fuse settings. Optional, nullopt means that the default HVPS fuse settings should be used.
      */
     std::optional<HVPSFuseSettings> hvpsFuseSettings;

     /**
      * User-provided callbacks keyed by us4OEM system interrupt. A callback
      * is registered with the underlying interrupt handler only for the keys
      * present in this map. Default: empty (no callbacks registered).
      */
     Us4OEMInterruptCallbacksMap interruptCallbacks;
};

/**
 * Builder for Us4RSettings.
 *
 * Lets callers (in particular, those that obtain a Us4RSettings from a
 * configuration file) produce a modified copy of an existing Us4RSettings
 * without having to re-pass every field through the public constructor.
 * Us4RSettings itself remains immutable.
 *
 * Initialize the builder with an existing Us4RSettings, call any of the
 * setters to override individual fields, then call build() to obtain the
 * resulting Us4RSettings.
 */
class Us4RSettingsBuilder {
public:
    using ReprogrammingMode = Us4OEMSettings::ReprogrammingMode;

    /** Initialize the builder with the values of an existing Us4RSettings. */
    explicit Us4RSettingsBuilder(const Us4RSettings &settings)
        : probeAdapterSettings(settings.getProbeAdapterSettings().value()),
          probeSettings(settings.getProbeSettingsList()),
          rxSettings(settings.getRxSettings().value()),
          hvSettings(settings.getHVSettings()),
          channelsMask(settings.getChannelsMaskForAllProbes()),
          reprogrammingMode(settings.getReprogrammingMode()),
          nUs4OEMs(settings.getNumberOfUs4oems()),
          adapterToUs4RModuleNumber(settings.getAdapterToUs4RModuleNumber()),
          externalTrigger(settings.isExternalTrigger()),
          txFrequencyRange(settings.getTxFrequencyRange()),
          digitalBackplaneSettings(settings.getDigitalBackplaneSettings()),
          bitstreams(settings.getBitstreams()),
          limits(settings.getTxRxLimits()),
          watchdogSettings(settings.getWatchdogSettings()),
          allowDuplicateOEMIds(settings.isAllowDuplicateOEMIds()),
          maskDVDDInterrupt(settings.isDVDDInterruptMasked()),
          hvpsFuseSettings(settings.getHVPSFuseSettings()),
          interruptCallbacks(settings.getInterruptCallbacks()) {}

    Us4RSettingsBuilder &setProbeAdapterSettings(ProbeAdapterSettings v) {
        probeAdapterSettings = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setProbeSettings(std::vector<ProbeSettings> v) {
        probeSettings = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setRxSettings(RxSettings v) {
        rxSettings = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setHVSettings(std::optional<HVSettings> v) {
        hvSettings = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setChannelsMask(std::vector<std::unordered_set<ChannelIdx>> v) {
        channelsMask = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setReprogrammingMode(ReprogrammingMode v) {
        reprogrammingMode = v; return *this;
    }
    Us4RSettingsBuilder &setNumberOfUs4oems(std::optional<Ordinal> v) {
        nUs4OEMs = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setAdapterToUs4RModuleNumber(std::vector<Ordinal> v) {
        adapterToUs4RModuleNumber = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setExternalTrigger(bool v) { externalTrigger = v; return *this; }
    Us4RSettingsBuilder &setTxFrequencyRange(int v) { txFrequencyRange = v; return *this; }
    Us4RSettingsBuilder &setDigitalBackplaneSettings(std::optional<DigitalBackplaneSettings> v) {
        digitalBackplaneSettings = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setBitstreams(std::vector<Bitstream> v) {
        bitstreams = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setTxRxLimits(std::optional<Us4RTxRxLimits> v) {
        limits = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setWatchdogSettings(WatchdogSettings v) {
        watchdogSettings = std::move(v); return *this;
    }
    Us4RSettingsBuilder &setAllowDuplicateOEMIds(bool v) { allowDuplicateOEMIds = v; return *this; }
    Us4RSettingsBuilder &setMaskDVDDInterrupt(bool v) { maskDVDDInterrupt = v; return *this; }
    Us4RSettingsBuilder &setHVPSFuseSettings(std::optional<HVPSFuseSettings> v) {
        hvpsFuseSettings = std::move(v); return *this;
    }
    /** Replaces the whole map of per-interrupt callbacks. */
    Us4RSettingsBuilder &setInterruptCallbacks(Us4OEMInterruptCallbacksMap v) {
        interruptCallbacks = std::move(v); return *this;
    }

    /** Sets the callback for a single interrupt; replaces any existing entry. */
    Us4RSettingsBuilder &setInterruptCallback(Us4OEMInterrupt interrupt,
                                              Us4OEMInterruptCallback callback) {
        interruptCallbacks[interrupt] = std::move(callback);
        return *this;
    }

    /**
     * Registers the same callback for every safe-state us4OEM interrupt.
     * The safe-state interrupt is the interrupt that is raised when
     * some exceptional issue occurred on the system (e.g. too long TX pulse was detected).
     * The callback receives both the interrupt that fired and the OEM ordinal,
     * so a single function can branch on the interrupt if needed.
     * Replaces any per-interrupt entries previously set on this builder.
     * Passing an empty std::function clears all five entries.
     */
    Us4RSettingsBuilder &setInterruptCallbackForAllSafeStateInterrupts(
        std::function<void(Us4OEMInterrupt, Ordinal)> callback) {
        constexpr Us4OEMInterrupt all[] = {
            Us4OEMInterrupt::PROBE_NOT_CONNECTED,
            Us4OEMInterrupt::PULSER_INTERRUPT,
            Us4OEMInterrupt::TX_TIMEOUT,
            Us4OEMInterrupt::WATCHDOG_IRQ1,
            Us4OEMInterrupt::HVPS_FUSE
        };
        if(!callback) {
            for(auto irq : all) {
                interruptCallbacks[irq] = {};
            }
            return *this;
        }
        for(auto irq : all) {
            interruptCallbacks[irq] = [callback, irq](Ordinal oem) {
                callback(irq, oem);
            };
        }
        return *this;
    }

    Us4RSettings build() const {
        return Us4RSettings(
            probeAdapterSettings,
            probeSettings,
            rxSettings,
            hvSettings,
            channelsMask,
            reprogrammingMode,
            nUs4OEMs,
            adapterToUs4RModuleNumber,
            externalTrigger,
            txFrequencyRange,
            digitalBackplaneSettings,
            bitstreams,
            limits,
            watchdogSettings,
            allowDuplicateOEMIds,
            maskDVDDInterrupt,
            hvpsFuseSettings,
            interruptCallbacks);
    }

private:
    ProbeAdapterSettings probeAdapterSettings;
    std::vector<ProbeSettings> probeSettings;
    RxSettings rxSettings;
    std::optional<HVSettings> hvSettings;
    std::vector<std::unordered_set<ChannelIdx>> channelsMask;
    ReprogrammingMode reprogrammingMode;
    std::optional<Ordinal> nUs4OEMs;
    std::vector<Ordinal> adapterToUs4RModuleNumber;
    bool externalTrigger;
    int txFrequencyRange;
    std::optional<DigitalBackplaneSettings> digitalBackplaneSettings;
    std::vector<Bitstream> bitstreams;
    std::optional<Us4RTxRxLimits> limits;
    WatchdogSettings watchdogSettings;
    bool allowDuplicateOEMIds;
    bool maskDVDDInterrupt;
    std::optional<HVPSFuseSettings> hvpsFuseSettings;
    Us4OEMInterruptCallbacksMap interruptCallbacks;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H
