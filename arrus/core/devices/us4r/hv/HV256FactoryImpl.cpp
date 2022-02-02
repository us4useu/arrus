#include "HV256FactoryImpl.h"

#include <ius4oem.h>
#include <idbarLite.h>
#include <ihv256.h>

#include "arrus/core/devices/us4r/external/ius4oem/Us4RLoggerWrapper.h"

namespace arrus::devices {


HV256Impl::Handle HV256FactoryImpl::getHV256(const HVSettings &settings, IUs4OEM *master) {
    std::unique_ptr<IDBAR> dbarLite(GetDBARLite(dynamic_cast<II2CMaster*>(master)));
    Logger::SharedHandle arrusLogger = getLoggerFactory()->getLogger();
    us4r::Logger::Handle logger =
        std::make_unique<Us4RLoggerWrapper>(arrusLogger);

    std::unique_ptr<IHV> hv256(GetHV256(dbarLite->GetI2CHV(), std::move(logger)));
    DeviceId id(DeviceType::HV, 0);

    return std::make_unique<HV256Impl>(id, settings.getModelId(),
                                       std::move(dbarLite), std::move(hv256));
}
}