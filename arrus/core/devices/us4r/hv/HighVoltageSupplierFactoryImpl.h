#ifndef ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H

#include "HighVoltageSupplierFactory.h"

namespace arrus::devices {

class HighVoltageSupplierFactoryImpl : public HighVoltageSupplierFactory {
public:
    HighVoltageSupplier::Handle getHighVoltageSupplier(const HVSettings &settings, IUs4OEM *master) override;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H
