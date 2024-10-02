#ifndef ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H
#define ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H

#include <memory>

#include <idbar.h>
#include <ihv.h>
#include <us4rExceptions.h>

#include "arrus/common/format.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/us4r/HVModelId.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/common/logging.h"

namespace arrus::devices {

/**
 * us4us HV interface.
 */
class HighVoltageSupplier : public Device {
public:
    using Handle = std::unique_ptr<HighVoltageSupplier>;

    HighVoltageSupplier(const DeviceId &id, HVModelId modelId);

    void setVoltage(const std::vector<HVVoltage> &voltages) {
        std::vector<IHVVoltage> us4RVoltages;
        std::transform(std::begin(voltages), std::end(voltages), std::back_insert_iterator(us4RVoltages),
                       [](const HVVoltage &v) { return IHVVoltage(v.getVoltageMinus(), v.getVoltagePlus()); });
        try {
            getIHV()->EnableHV();
            getIHV()->SetHVVoltage(us4RVoltages);
        } catch (const ::us4us::ValidationException &) {
            // Disable HV and Propage validation errors.
            try {
                getIHV()->DisableHV();
            } catch( const std::exception &ee) {
                logger->log(LogSeverity::ERROR, format("Exception while disabling HV: {}", ee.what()));
            }
            throw;
        } catch (const ::us4us::AssertionException &) {
            // Disable HV and Propage validation errors.
            try {
                getIHV()->DisableHV();
            } catch( const std::exception &ee) {
                logger->log(LogSeverity::ERROR, format("Exception while disabling HV: {}", ee.what()));
            }
            throw;
        } catch (const std::exception &e) {
            logger->log(LogSeverity::INFO,
                        ::arrus::format("First attempt to set HV voltage failed with "
                                        "message: '{}', trying once more.",
                                        e.what()));
            getIHV()->EnableHV();
            getIHV()->SetHVVoltage(us4RVoltages);
        }
    }

    unsigned char getVoltage() { return getIHV()->GetHVVoltage(); }

    float getMeasuredPVoltage() { return getIHV()->GetMeasuredHVPVoltage(); }

    float getMeasuredMVoltage() { return getIHV()->GetMeasuredHVMVoltage(); }

    void disable() {
        try {
            getIHV()->DisableHV();
        } catch (std::exception &e) {
            logger->log(LogSeverity::INFO,
                        ::arrus::format("First attempt to disable high voltage failed with "
                                        "message: '{}', trying once more.",
                                        e.what()));
            getIHV()->DisableHV();
        }
    }

    const HVModelId &getModelId() const { return modelId; }

protected:
    virtual IHV *getIHV() = 0;

private:
    Logger::Handle logger;
    HVModelId modelId;
};

/**
 * Us4us HV interface. This class owns the handle to the HV.
 */
class HighVoltageSupplierOwner : public HighVoltageSupplier {
public:
    HighVoltageSupplierOwner(const DeviceId &id, HVModelId modelId, std::unique_ptr<IHV> hv)
        : HighVoltageSupplier(id, std::move(modelId)), hv(std::move(hv)) {}

protected:
    IHV *getIHV() override { return hv.get(); }

private:
    std::unique_ptr<IHV> hv;
};

/**
 * Us4us HV interface. This class does not own the handle to the HV (it's only a view).
 */
class HighVoltageSupplierView : public HighVoltageSupplier {
public:
    HighVoltageSupplierView(const DeviceId &id, HVModelId modelId, IHV *hv)
        : HighVoltageSupplier(id, std::move(modelId)), hv(hv) {}

protected:
    IHV *getIHV() override { return hv; }

private:
    IHV *hv;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_HV_HV256IMPL_H
