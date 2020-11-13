#ifndef ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H

#include "HV256Factory.h"

namespace arrus::devices {

class HV256FactoryImpl : public HV256Factory {
public:
    HV256Impl::Handle getHV256(const HVSettings &settings, IUs4OEM *master) override;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HV_HV256FACTORYIMPL_H
