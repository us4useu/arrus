#ifndef ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H
#define ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H

#include <memory>

#include <idbarLite.h>
#include <ihv256.h>

#include "arrus/core/common/logging.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/us4r/HVModelId.h"
#include "arrus/common/format.h"

namespace arrus::devices {

class HV256Impl : public Device {
public:
    using Handle = std::unique_ptr<HV256Impl>;

    HV256Impl(const DeviceId &id, HVModelId modelId,
              std::unique_ptr<IDBARLite> dbarLite,
              std::unique_ptr<IHV256> hv256);

    void setVoltage(Voltage voltage) {
        try {
            hv256->EnableHV();
            hv256->SetHVVoltage(voltage);
        } catch(std::exception &e) {
            // TODO catch a specific exception
            logger->log(
                LogSeverity::INFO,
                ::arrus::format(
                    "First attempt to set HV voltage failed with "
                    "message: '{}', trying once more.",
                    e.what()));
            hv256->EnableHV();
            hv256->SetHVVoltage(voltage);
        }
    }

    void disable() {
        try {
            hv256->DisableHV();
        } catch(std::exception &e) {
            logger->log(LogSeverity::INFO,
                        ::arrus::format(
                            "First attempt to disable high voltage failed with "
                            "message: '{}', trying once more.",
                            e.what()));
            hv256->DisableHV();
        }
    }

private:
    Logger::Handle logger;
    HVModelId modelId;
    std::unique_ptr<IDBARLite> dbarLite;
    std::unique_ptr<IHV256> hv256;
};


}

#endif //ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H
