#ifndef ARRUS_CORE_DEVICES_US4R_BACKPLANE_DIGITALBACKPLANEFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_BACKPLANE_DIGITALBACKPLANEFACTORYIMPL_H

#include <idbar.h>
#include <idbarLite.h>
#include <idbarLitePcie.h>
#include <iUs4RDBAR.h>

#include "DigitalBackplane.h"
#include "DigitalBackplaneFactory.h"

namespace arrus::devices {

class DigitalBackplaneFactoryImpl: public DigitalBackplaneFactory {
public:
    std::optional<DigitalBackplane::Handle> getDigitalBackplane(const HVSettings &settings,
                                                                const std::vector<IUs4OEM *> &us4oems) override {
        const std::string &manufacturer = settings.getModelId().getManufacturer();
        const std::string &name = settings.getModelId().getName();
        if(name == "hv256")  {
            auto ver = us4oems[0]->GetOemVersion();
            if(ver == 1) {
                std::unique_ptr<IDBAR> dbar(GetDBARLite(dynamic_cast<II2CMaster *>(us4oems[0])));
                return std::make_unique<DigitalBackplane>(std::move(dbar));
            }
            else if(ver == 2) {
                std::unique_ptr<IDBAR> dbar(GetDBARLitePcie(dynamic_cast<II2CMaster *>(us4oems[0])));
                return std::make_unique<DigitalBackplane>(std::move(dbar));
            }
            else {
                throw std::runtime_error("Invalid OEM version");
            }
        }
        else if(name == "us4rpsc") {
            std::unique_ptr<IDBAR> dbar(GetUs4RDBAR(dynamic_cast<II2CMaster *>(us4oems[0])));
            return std::make_unique<DigitalBackplane>(std::move(dbar));
        }
        else if(name == "us4oemhvps") {
            // TODO consider allowing DBAR even if us4oemhvps is used.
            return std::nullopt;
        }
        else {
            throw IllegalArgumentException(
                ::arrus::format("Unrecognized high-voltage supplier: {}, {}", manufacturer, name));
        }
    }
};

}

#endif//ARRUS_CORE_DEVICES_US4R_BACKPLANE_DIGITALBACKPLANEFACTORYIMPL_H
