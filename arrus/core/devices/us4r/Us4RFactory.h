#ifndef ARRUS_CORE_DEVICES_US4R_US4RFACTORY_H
#define ARRUS_CORE_DEVICES_US4R_US4RFACTORY_H

#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/devices/us4r/Us4RSettings.h"

namespace arrus::devices {

class Us4RFactory {
public:
    using Handle = std::unique_ptr<Us4RFactory>;

    virtual Us4R::Handle getUs4R(Ordinal ordinal, const Us4RSettings &settings) = 0;

    virtual ~Us4RFactory() = default;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RFACTORY_H
