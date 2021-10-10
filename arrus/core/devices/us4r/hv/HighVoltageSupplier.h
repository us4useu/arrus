#ifndef ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H
#define ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H

#include <memory>

#include <idbar.h>
#include <ihv.h>

#include "arrus/core/common/logging.h"
#include "arrus/common/format.h"
#include "arrus/core/api/devices/us4r/HVModelId.h"
#include "arrus/core/api/devices/Device.h"


namespace arrus::devices {

class HighVoltageSupplier : public Device {
public:
    using Handle = std::unique_ptr<HighVoltageSupplier>;

    HighVoltageSupplier(const DeviceId &id, HVModelId modelId,
                        std::unique_ptr<IDBAR> dbar,
                        std::unique_ptr<IHV> hv);

    void setVoltage(Voltage voltage) {
        try {
            hv->EnableHV();
            hv->SetHVVoltage(voltage);
        } catch(std::exception &e) {
            // TODO catch a specific exception
            logger->log(
                LogSeverity::INFO,
                ::arrus::format(
                    "First attempt to set HV voltage failed with "
                    "message: '{}', trying once more.",
                    e.what()));
            hv->EnableHV();
            hv->SetHVVoltage(voltage);
        }
    }

    void disable() {
        try {
            hv->DisableHV();
        } catch(std::exception &e) {
            logger->log(LogSeverity::INFO,
                        ::arrus::format(
                            "First attempt to disable high voltage failed with "
                            "message: '{}', trying once more.",
                            e.what()));
            hv->DisableHV();
        }
    }

private:
    Logger::Handle logger;
    HVModelId modelId;
    std::unique_ptr<IDBAR> dbar;
    std::unique_ptr<IHV> hv;
};


}

#endif //ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H
