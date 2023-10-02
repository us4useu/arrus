#include "HighVoltageSupplierFactoryImpl.h"

#include <ius4oem.h>
#include <idbar.h>
#include <ihv.h>
#include <iUs4RDBAR.h>
#include <iUs4RPSC.h>
#include <idbarLite.h>
#include <idbarLitePcie.h>
#include <ihv256.h>

#include "arrus/common/asserts.h"
#include "arrus/core/devices/us4r/external/ius4oem/Us4RLoggerWrapper.h"

namespace arrus::devices {

std::vector<HighVoltageSupplier::Handle>
HighVoltageSupplierFactoryImpl::getHighVoltageSupplier(
    const HVSettings &settings, const std::vector<IUs4OEM*> &us4oems,
    const std::optional<DigitalBackplane::Handle> &backplane
    ) {

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

        std::vector<HighVoltageSupplier::Handle> hvs;
        auto ver = us4oems[0]->GetOemVersion();
        ARRUS_REQUIRES_TRUE(backplane.has_value(), "Backplane is required for hv256.");
        IDBAR* dbar = backplane.value()->getIDBAR();

        if(ver == 1) {
            std::unique_ptr<IHV> hv(GetHV256(dbar->GetI2CHV(), std::move(logger)));
            auto _hv = std::make_unique<HighVoltageSupplier>(id, settings.getModelId(), std::move(hv));
            hvs.push_back(std::move(_hv));
        }
        else if(ver == 2) {
            std::unique_ptr<IHV> hv(GetHV256(dbar->GetI2CHV(), std::move(logger)));
            auto _hv = std::make_unique<HighVoltageSupplier>(id, settings.getModelId(), std::move(hv));
            hvs.push_back(std::move(_hv));
        }
        else {
            throw std::runtime_error("Invalid OEM version");
        }

        return hvs;
    }
    else if(name == "us4rpsc") {
        std::vector<HighVoltageSupplier::Handle> hvs;
        ARRUS_REQUIRES_TRUE(backplane.has_value(), "Backplane is required for hv256.");
        IDBAR* dbar = backplane.value()->getIDBAR();
        std::unique_ptr<IHV> hv(GetUs4RPSC(dbar->GetI2CHV(), std::move(logger)));
        auto _hv =  std::make_unique<HighVoltageSupplier>(id, settings.getModelId(), std::move(hv));
        hvs.push_back(std::move(_hv));

        return hvs;
    }
    else if(name == "us4oemhvps") {
        std::vector<HighVoltageSupplier::Handle> hvs;

        for(auto us4oem: us4oems) {
            std::unique_ptr<IHV> hv(us4oem->getHVPS());
            hvs.push_back(std::make_unique<HighVoltageSupplier>(id, settings.getModelId(), std::move(hv)));
        }

        return hvs;

    }
    else {
        throw IllegalArgumentException(
            ::arrus::format("Unrecognized high-voltage supplier: {}, {}", manufacturer, name));
    }
}
}
