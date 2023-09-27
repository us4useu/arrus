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

std::vector<HighVoltageSupplier::Handle>
HighVoltageSupplierFactoryImpl::getHighVoltageSupplier(const HVSettings &settings, const std::vector<IUs4OEM*> &us4oems) {
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
        std::unique_ptr<IDBAR> dbar(GetDBARLite(dynamic_cast<II2CMaster *>(us4oems[0])));
        std::unique_ptr<IHV> hv(GetHV256(dbar->GetI2CHV(), std::move(logger)));
        return std::make_unique<HighVoltageSupplier>(id, settings.getModelId(), std::move(dbar), std::move(hv));
    }
    else if(name == "hv256p")  {
        // TODO how to detect if we have DBAR-Lite PCIE or the legacy DBAR-Lite?
        std::unique_ptr<IDBAR> dbar(GetDBARLitePcie(dynamic_cast<II2CMaster *>(us4oems[0])));
        std::unique_ptr<IHV> hv(GetHV256(dbar->GetI2CHV(), std::move(logger)));
        return std::vector<HighVoltageSupplier>(id, settings.getModelId(), std::move(dbar), std::move(hv));
    }
    else if(name == "us4rpsc") {
        // TODO?
        throw std::runtime_error("unsupported: " + name + " please set hv: hv256-pcie in the prototxt file");
//        std::unique_ptr<IHV> hv(GetUs4RPSC(dbar->GetI2CHV(), std::move(logger)));
//        return std::make_unique<HighVoltageSupplier>(id, settings.getModelId(), std::move(dbar), std::move(hv));
    }
    else if(name == "us4oemhvps") {
        std::vector<HighVoltageSupplier::Handle> hvs(us4oems.size(), nullptr);
        std::transform(std::begin(us4oems), std::end(us4oems), std::begin(hvs), [](IUs4OEM* us4oem){
            return std::make_unique<HighVoltageSupplier::Handle>(us4oem->getHVPS());
        }); 
        return hvs;
    }
    else {
        throw IllegalArgumentException(
            ::arrus::format("Unrecognized high-voltage supplier: {}", manufacturer));
    }
}
}