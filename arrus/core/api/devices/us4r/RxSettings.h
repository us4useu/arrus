#ifndef ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H

#include <Us4OEM/api/RxSettings.h>

namespace arrus::devices {

// Backward compatibility
using RxSettings = ::us4us::us4r::RxSettings;
using RxSettingsBuilder = ::us4us::us4r::RxSettingsBuilder;

}

#endif //ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H
