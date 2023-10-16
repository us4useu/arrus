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

/**
 * us4us HV interface.
 */
class HighVoltageSupplier : public Device {
public:
    using Handle = std::unique_ptr<HighVoltageSupplier>;

    HighVoltageSupplier(const DeviceId &id, HVModelId modelId);

    void setVoltage(Voltage voltage) {
        try {
            getIHV()->EnableHV();
            getIHV()->SetHVVoltage(voltage);
        } catch(std::exception &e) {
            // TODO catch a specific exception
            logger->log(
                LogSeverity::INFO,
                ::arrus::format(
                    "First attempt to set HV voltage failed with "
                    "message: '{}', trying once more.",
                    e.what()));
            getIHV()->EnableHV();
            getIHV()->SetHVVoltage(voltage);
        }
    }

    unsigned char getVoltage() {
        return getIHV()->GetHVVoltage();
    }

    float getMeasuredPVoltage() {
        return getIHV()->GetMeasuredHVPVoltage();
    }

    float getMeasuredMVoltage() {
        return getIHV()->GetMeasuredHVMVoltage();
    }

    void disable() {
        try {
            getIHV()->DisableHV();
        } catch(std::exception &e) {
            logger->log(LogSeverity::INFO,
                        ::arrus::format(
                            "First attempt to disable high voltage failed with "
                            "message: '{}', trying once more.",
                            e.what()));
            getIHV()->DisableHV();
        }
    }

    const HVModelId &getModelId() const {
        return modelId;
    }

protected:
    virtual IHV* getIHV() = 0;

private:
    Logger::Handle logger;
    HVModelId modelId;
};

/**
 * Us4us HV interface. This class owns the handle to the HV.
 */
class HighVoltageSupplierOwner: public HighVoltageSupplier {
public:
    HighVoltageSupplierOwner(const DeviceId &id, HVModelId modelId, std::unique_ptr<IHV> hv)
        : HighVoltageSupplier(id, std::move(modelId), hv(std::move(hv)) {}

protected:
    IHV *getIHV() override { return hv.get(); }

private:
    std::unique_ptr<IHV> hv;
};

/**
 * Us4us HV interface. This class does not own the handle to the HV (it's only a view).
 */
class HighVoltageSupplierView: public HighVoltageSupplier {
public:
    HighVoltageSupplierView(const DeviceId &id, HVModelId modelId, IHV *hv)
        : HighVoltageSupplier(id, std::move(modelId), std::move(dbar)), hv(hv) {}

protected:
    IHV *getIHV() override { return hv; }

private:
    IHV* hv;
};


}

#endif //ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H
