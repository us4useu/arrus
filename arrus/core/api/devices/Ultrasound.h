#ifndef ARRUS_CORE_API_DEVICES_ULTRASOUND_H
#define ARRUS_CORE_API_DEVICES_ULTRASOUND_H

#include <memory>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
// TODO(pjarosik) avoid using us4r specific objects here
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/session/Metadata.h"

namespace arrus::devices {

/**
 * An interface to the ultrasound device.
 */
class Ultrasound : public Device {
public:
    using Handle = std::unique_ptr<Ultrasound>;

    explicit Ultrasound(const DeviceId &id) : Device(id) {}

    ~Ultrasound() override = default;

    virtual std::pair<framework::Buffer::SharedHandle, std::vector<session::Metadata::SharedHandle>>
    upload(const ::arrus::ops::us4r::Scheme &scheme) = 0;

    virtual void start() = 0;
    virtual void stop() = 0;
    /**
     * Trigger a single run of the current work mode (TX/RX in case of workMode=MANUAL_OP,
     * sequence of TX/RXs in other cases).
     *
     * @param sync whether this method should work in a synchronous or asynchronous; true means synchronous, i.e.
     *        the caller will wait until the triggered TX/RX or sequence of TX/RXs has been done.
     * @param timeout timeout [ms]; std::nullopt means to wait infinitely. This parameter is only relevant when
     *        sync = true.
     */
    virtual void trigger(bool sync = false, std::optional<long long> timeout = std::nullopt) = 0;


    /**
     * Synchronization point with us4R system. After returning from this method, the last "TX/RX" (triggered by the
     * trigger method will be  fully executed by the system.
     *
     * Sync with "SEQ_IRQ" interrupt (i.e. wait until the SEQ IRQ will occur).
     *
     * @param timeout timeout in number of milliseconds
     */
    virtual void sync(std::optional<long long> timeout) = 0;

    /**
     * Returns NOMINAL Ultrasound device sampling frequency.
     */
    virtual float getSamplingFrequency() const = 0;

    /**
     * Returns the sampling frequency with which data from us4R will be acquired. The returned value
     * depends on the result of sequence upload (e.g. DDC decimation factor).
     */
    virtual float getCurrentSamplingFrequency() const = 0;

    /**
     * Returns probe identified by given ordinal number.
     *
     * @param ordinal ordinal number of the probe to get
     * @return probe handle
     */
    virtual Probe *getProbe(Ordinal ordinal) = 0;

    /**
     * Returns the number of probes that are connected to the system.
     */
    virtual int getNumberOfProbes() const = 0;

    virtual std::pair<std::shared_ptr<framework::Buffer>, std::shared_ptr<session::Metadata>>
    setSubsequence(SequenceId sequenceId, uint16 start, uint16 end, const std::optional<float> &sri) = 0;

    Ultrasound(Ultrasound const &) = delete;
    Ultrasound(Ultrasound const &&) = delete;
    void operator=(Ultrasound const &) = delete;
    void operator=(Ultrasound const &&) = delete;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_ULTRASOUND_H
