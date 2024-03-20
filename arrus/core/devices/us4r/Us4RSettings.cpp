#include "Us4RSettings.h"

#include "arrus/core/devices/probe/ProbeSettings.h"
#include "arrus/core/devices/us4r/RxSettings.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterSettings.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMSettings.h"

namespace arrus::devices {

template<typename T> static inline void printOptionalValue(const std::optional<T> value, std::ostream &os) {
    if (value.has_value()) {
        os << value.value();
    } else {
        os << "(no value)";
    };
}

std::ostream &operator<<(std::ostream &os, const Us4RSettings &settings) {
    os << "us4oemSettings: ";
    int i = 0;
    for (const auto &us4oemSetting : settings.getUs4OEMSettings()) {
        os << "Us4OEM:" << i++ << ": ";
        os << us4oemSetting << "; ";
    }

    auto &probeAdapterSettings = settings.getProbeAdapterSettings();
    auto &probeSettings = settings.getProbeSettingsList();
    auto &rxSettings = settings.getRxSettings();
    auto &channelsMasks = settings.getChannelsMaskForAllProbes();
    auto &us4oemChannelsMasks = settings.getUs4OEMChannelsMask();

    os << " probeAdapterSettings: ";
    printOptionalValue(probeAdapterSettings, os);
    for (auto &probe : probeSettings) {
        os << " probeSettings: ";
        os << probe;
    }
    os << " rxSettings: ";
    printOptionalValue(rxSettings, os);

    for (auto mask : channelsMasks) {
        os << " channels mask: ";
        for (auto ch : mask) {
            os << (int)ch << ", ";
        }
    }

    os << " us4oem channels mask: ";
    i = 0;
    for (const auto &vec : us4oemChannelsMasks) {
        os << "us4oem " << i << ": ";
        for (auto channel : vec) {
            os << (int) channel << ", ";
        }
        ++i;
    }
    return os;
}

}// namespace arrus::devices
