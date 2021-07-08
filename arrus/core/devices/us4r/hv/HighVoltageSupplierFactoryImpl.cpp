#include "HighVoltageSupplierFactoryImpl.h"

#include <ius4oem.h>
#include <idbar.h>
#include <ihv.h>
#include <iUs4RDBAR.h>
#include <iUs4RPSC.h>
#include <idbarLite.h>
#include <ihv256.h>

#include "arrus/common/asserts.h"
#include "arrus/core/devices/us4r/external/ius4oem/Us4RLoggerWrapper.h"

namespace arrus::devices {

HighVoltageSupplier::Handle
HighVoltageSupplierFactoryImpl::getHighVoltageSupplier(const HVSettings &settings, IUs4OEM *master) {
    const std::string &manufacturer = settings.getModelId().getManufacturer();
    const std::string &name = settings.getModelId().getName();
    ARRUS_REQUIRES_EQUAL(manufacturer, "us4us",
                         IllegalArgumentException(
                             ::arrus::format(
                                 "Only us4us High-Voltage suppliers are supported only (got {})",
                                 manufacturer)));
    Logger::SharedHandle arrusLogger = getLoggerFactory()->getLogger();
    us4r::Logger::Handle logger = std::make_unique<Us4RLoggerWrapper>(arrusLogger);
    DeviceId id(DeviceType::HV, 0);

    if(name == "hv256")  {
        std::unique_ptr<IDBAR> dbar(GetDBARLite(reinterpret_cast<II2CMaster *>(master)));
        std::unique_ptr<IHV> hv(GetHV256(dbar->GetI2CHV(), std::move(logger)));
        return std::make_unique<HighVoltageSupplier>(id, settings.getModelId(), std::move(dbar), std::move(hv));
    }
    else if(name == "us4rpsc") {
        std::unique_ptr<IDBAR> dbar(GetUs4RDBAR(reinterpret_cast<II2CMaster *>(master)));
        std::unique_ptr<IHV> hv(GetUs4RPSC(dbar->GetI2CHV(), std::move(logger)));
        return std::make_unique<HighVoltageSupplier>(id, settings.getModelId(), std::move(dbar), std::move(hv));
    }
    else {
        throw IllegalArgumentException(
            ::arrus::format("Unrecognized high-voltage supplier: {}", manufacturer));
    }
}
}