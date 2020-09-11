#include "Us4RSettings.h"

#include "arrus/core/devices/us4r/us4oem/Us4OEMSettings.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterSettings.h"
#include "arrus/core/devices/probe/ProbeSettings.h"
#include "arrus/core/devices/us4r/RxSettings.h"

namespace arrus::devices {

template<typename T>
static inline void printOptionalValue(const std::optional<T> value,
                                 std::ostream &os) {
    if(value.has_value()) {
        os << value.value();
    } else {
        os << "(no value)";
    };
}

std::ostream &
operator<<(std::ostream &os, const Us4RSettings &settings) {
    os << "us4oemSettings: ";
    int i = 0;
    for(auto us4oemSetting : settings.getUs4OEMSettings()) {
        os << "Us4OEM:" << i++ << ": ";
        os << us4oemSetting << "; ";
    }

    auto &probeAdapterSettings = settings.getProbeAdapterSettings();
    auto &probeSettings = settings.getProbeSettings();
    auto &rxSettings = settings.getRxSettings();

    os << " probeAdapterSettings: ";
    printOptionalValue(probeAdapterSettings, os);
    os << " probeSettings: ";
    printOptionalValue(probeSettings, os);
    os << " rxSettings: ";
    printOptionalValue(rxSettings, os);
    return os;
}

}


