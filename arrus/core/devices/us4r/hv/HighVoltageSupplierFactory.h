#ifndef ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORY_H
#define ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORY_H


#include "arrus/core/api/devices/us4r/HVSettings.h"
#include "arrus/core/devices/us4r/hv/HighVoltageSupplier.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/devices/us4r/backplane/DigitalBackplane.h"

namespace arrus::devices {

class HighVoltageSupplierFactory {
public:
    virtual std::vector<HighVoltageSupplier::Handle> getHighVoltageSupplier(
        const HVSettings &settings, const std::vector<IUs4OEM *> &us4oems,
        const std::optional<DigitalBackplane::Handle> &backplane
        ) = 0;

    virtual ~HighVoltageSupplierFactory() = default;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORY_H
