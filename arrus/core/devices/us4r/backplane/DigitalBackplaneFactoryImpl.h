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

    std::optional<DigitalBackplane::Handle> getDigitalBackplane(const DigitalBackplaneSettings &settings,
                                                                const std::vector<IUs4OEM *> &us4oems) override {
        const std::string &manufacturer = settings.getModelId().getManufacturer();
        const std::string &name = settings.getModelId().getName();
        if(manufacturer != "us4us") {
            throw IllegalArgumentException("Only 'us4us' digital backplane is supported.");
        }
        if(name == "dbarlite")  {
            return getDBARLite(us4oems, 32);
        }
        if(name == "dbarlite_8bit")  {
            return getDBARLite(us4oems, 8);
        }
        else if(name == "us4rdbar") {
            std::unique_ptr<IDBAR> dbar(GetUs4RDBAR(dynamic_cast<II2CMaster *>(us4oems[0])));
            return std::make_unique<DigitalBackplane>(std::move(dbar));
        }
        else {
            throw IllegalArgumentException(
                ::arrus::format("Unrecognized DBAR: {}, {}", manufacturer, name));
        }
    }

    std::optional<DigitalBackplane::Handle> getDigitalBackplane(const HVSettings &settings,
                                                                const std::vector<IUs4OEM *> &us4oems) override {
        const std::string &manufacturer = settings.getModelId().getManufacturer();
        const std::string &name = settings.getModelId().getName();
        if(name == "hv256")  {
            return getDBARLite(us4oems, 32);
        }
        else if(name == "us4rpsc") {
            std::unique_ptr<IDBAR> dbar(GetUs4RDBAR(dynamic_cast<II2CMaster *>(us4oems[0])));
            return std::make_unique<DigitalBackplane>(std::move(dbar));
        }
        else if(name == "us4oemhvps") {
            return std::nullopt;
        }
        else {
            throw IllegalArgumentException(
                ::arrus::format("Unrecognized high-voltage supplier: {}, {}", manufacturer, name));
        }
    }

    std::optional<DigitalBackplane::Handle> getDBARLite(const std::vector<IUs4OEM *> &us4oems, uint8_t addrMode) const {
        auto ver = us4oems[0]->GetOemVersion();
        if(ver == 1) {
            std::unique_ptr<IDBAR> dbar(GetDBARLite(dynamic_cast<II2CMaster *>(us4oems[0]), addrMode));
            return std::make_unique<DigitalBackplane>(std::move(dbar));
        }
        else if(ver == 2) {
            if(addrMode != 32) {
                throw IllegalArgumentException("The DBARLite rev2 supports 32-bit addressing mode only.");
            }
            std::unique_ptr<IDBAR> dbar(GetDBARLitePcie(dynamic_cast<II2CMaster *>(us4oems[0])));
            return std::make_unique<DigitalBackplane>(std::move(dbar));
        }
        else {
            throw std::runtime_error("Invalid OEM version");
        }
    }
};

}

#endif//ARRUS_CORE_DEVICES_US4R_BACKPLANE_DIGITALBACKPLANEFACTORYIMPL_H
