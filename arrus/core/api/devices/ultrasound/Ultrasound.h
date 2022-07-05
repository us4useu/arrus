#ifndef ARRUS_CORE_API_DEVICES_ULTRASOUND_ULTRASOUND_H
#define ARRUS_CORE_API_DEVICES_ULTRASOUND_ULTRASOUND_H

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "arrus/core/api/framework/Metadata.h"

#include <memory>
#include <vector>

namespace arrus::devices {

/**
 * Ultrasound system.
 */
class Ultrasound: public Device {
public:
    using Handle = std::unique_ptr<Ultrasound>;

    explicit Ultrasound(const DeviceId &id): Device(id) {}

    ~Ultrasound() override = default;

    virtual std::pair<std::shared_ptr<arrus::framework::Buffer>, ::arrus::framework::Metadata>
    upload(const ::arrus::ops::us4r::TxRxSequence &seq, unsigned short rxBufferSize,
           const ::arrus::ops::us4r::Scheme::WorkMode &workMode,
           const ::arrus::framework::DataBufferSpec &hostBufferSpec) = 0;

    /**
     * Sets HV voltage.
     *
     * @param voltage voltage to set [V]
     */
    virtual void setVoltage(Voltage voltage) = 0;

    /*
     * Starts the execution of this ultrasound system.
     */
    virtual void start() = 0;

    /*
     * Stops the execution of this ultrasound system.
     */
    virtual void stop() = 0;

    /**
     * Triggers a single execution of sequence.
     */
    virtual void trigger() = 0;

    /**
     * Returns us4R device sampling frequency.
     */
    virtual float getSamplingFrequency() const = 0;

    Ultrasound(Ultrasound const&) = delete;
    Ultrasound(Ultrasound const&&) = delete;
    void operator=(Ultrasound const&) = delete;
    void operator=(Ultrasound const&&) = delete;

protected:
    explicit Ultrasound(const DeviceId &id): Device(id) {}

};;

}

#endif//ARRUS_CORE_API_DEVICES_ULTRASOUND_ULTRASOUND_H
