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

    WatchdogSettings(float oemThreshold0, float oemThreshold1, float hostThreshold)
        : oemThreshold0(oemThreshold0), oemThreshold1(oemThreshold1), hostThreshold(hostThreshold) {}

    bool isEnabled() const { return enabled; }

    float getOEMThreshold0() const { return oemThreshold0; }
    float getOEMThreshold1() const { return oemThreshold1; }
    float getHostThreshold() const { return hostThreshold; }

private:
    WatchdogSettings() {}

    bool enabled{false};
    /** thresholds, that should be respected by the watchdog implemented in OEM [seconds] */
    float oemThreshold0{0};
    float oemThreshold1{0};
    /** threshold, that should be respected by the watchdog implemented in host PC [seconds] */
    float hostThreshold{0};
};


}

#endif//ARRUS_CORE_API_DEVICES_US4R_WATCHDOG_SETTINGS_H
