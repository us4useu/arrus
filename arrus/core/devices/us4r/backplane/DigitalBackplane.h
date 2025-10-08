#ifndef ARRUS_CORE_DEVICES_BACKPLANE_US4R_DIGITALBACKPLANE_H
#define ARRUS_CORE_DEVICES_BACKPLANE_US4R_DIGITALBACKPLANE_H

#include <idbar.h>
#include "arrus/common/cache.h"

namespace arrus::devices {
class DigitalBackplane {
public:
    using Handle = std::unique_ptr<DigitalBackplane>;

    explicit DigitalBackplane(std::unique_ptr<IDBAR> dbar)
        : dbar(std::move(dbar)),
          serialNumber([this](){return this->dbar->GetSerialNumber();}),
          revision([this](){return this->dbar->GetRevisionNumber();}),
          firmwareVersion([this]() {return this->dbar->GetFirmwareVersionString();})
    {}

    const char* getSerialNumber() {
        return serialNumber.get().c_str();
    }

    const char* getRevisionNumber() {
        return revision.get().c_str();
    }

    const char* getFirmwareVersion() {
        return firmwareVersion.get().c_str();
    }

    void enableExternalTrigger() {
        dbar->EnableExternalTrigger();
    }

    void enableInternalTrigger() {
        dbar->EnableInternalTrigger();
    }

    /**
     * TODO NOTE: this method should not be exposed in API.
     */
    IDBAR* getIDBAR() {
        return dbar.get();
    }

private:
    std::unique_ptr<IDBAR> dbar;
    arrus::Cached<std::string> serialNumber;
    arrus::Cached<std::string> revision;
    arrus::Cached<std::string> firmwareVersion;
};
}

#endif//ARRUS_CORE_DEVICES_BACKPLANE_US4R_DIGITALBACKPLANE_H
