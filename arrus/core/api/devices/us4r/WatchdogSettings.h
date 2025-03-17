#ifndef ARRUS_CORE_API_DEVICES_US4R_WATCHDOG_SETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_WATCHDOG_SETTINGS_H

#include "arrus/core/api/common/types.h"


namespace arrus::devices {

class WatchdogSettings {
public:

    static WatchdogSettings disabled() {
        WatchdogSettings settings;
        settings.enabled = false;
        return settings;
    }

    static WatchdogSettings defaultSettings() {
        return WatchdogSettings{1.0f, 1.1f, 1.0f};
    }

    WatchdogSettings(float oemThreshold0, float oemThreshold1, float hostThreshold)
        : enabled(true), oemThreshold0(oemThreshold0), oemThreshold1(oemThreshold1), hostThreshold(hostThreshold) {}

    bool isEnabled() const { return enabled; }

    float getOEMThreshold0() const { return oemThreshold0; }
    float getOEMThreshold1() const { return oemThreshold1; }
    float getHostThreshold() const { return hostThreshold; }

private:
    WatchdogSettings()  = default;
    bool enabled{false};
    /** thresholds, that should be respected by the watchdog implemented in OEM [seconds] */
    float oemThreshold0{0.0f};
    float oemThreshold1{0.0f};
    /** threshold, that should be respected by the watchdog implemented in host PC [seconds] */
    float hostThreshold{0.0f};
};


}

#endif//ARRUS_CORE_API_DEVICES_US4R_WATCHDOG_SETTINGS_H
