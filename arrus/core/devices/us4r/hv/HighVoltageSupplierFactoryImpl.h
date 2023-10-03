#ifndef ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H

#include "HighVoltageSupplierFactory.h"

namespace arrus::devices {

class HighVoltageSupplierFactoryImpl : public HighVoltageSupplierFactory {
public:
    std::vector<HighVoltageSupplier::Handle> getHighVoltageSupplier(const HVSettings &settings, const std::vector<IUs4OEM *> &master) override;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H
