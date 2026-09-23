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
#include "arrus/core/api/common/Slice.h"

namespace arrus::devices {

/**
 * An interface to the ultrasound device.
 */
class Ultrasound : public Device {
public:
    using Handle = std::unique_ptr<Ultrasound>;

    explicit Ultrasound(const DeviceId &id) : Device(id) {}

    ~Ultrasound() override = default;

    std::string getDescription() const override {
        return "Ultrasound device";
    }

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

    /**
     * Selects [start, end) slices for each sub-sequence.
     *
     * The `slices` array should have exactly n elements, where n is the number of currently uploaded sequences.
     * The element slice[i] sets the [start, end) range for the i-th sequence.
     *
     * The `sris` should have eactly n elements, or should be empty (which means that no additional sri should be
     * applied).
     *
     * To turn off the given sequence, just set start equal to end (e.g. Slice(0, 0)). For such sequences, the metadata
     * will describe only empty data (dummy metadata).
     *
     * @param slices slices to set to each Scheme sub-sequence
     * @param sris sris to apply to each Scheme sub-sequence
     * @return returns the buffer and metadata for the modified Scheme. The metadata array size is always equal to
     *   the number of seqeuences in the original Scheme
     */
    virtual std::pair<std::shared_ptr<framework::Buffer>, std::vector<std::shared_ptr<session::Metadata>>>
    setSubsequences(const std::vector<Slice> &slices, const std::vector<std::optional<float>> &sris) = 0;

    /**
     * Selects the given list of TX/RXs for each of the uploaded TX/RX sequences.
     *
     * This is a generalization of the method above: the selected TX/RXs do not have to be consecutive.
     * An empty list means that the given sequence should be turned off.
     *
     * @param ops the list of TX/RXs (ordinal numbers, increasing) to run, for each Scheme sub-sequence
     * @param sris sris to apply to each Scheme sub-sequence
     */
    virtual std::pair<std::shared_ptr<framework::Buffer>, std::vector<std::shared_ptr<session::Metadata>>>
    setSubsequences(const std::vector<std::vector<uint16>> &ops, const std::vector<std::optional<float>> &sris) = 0;

    /**
     * Prepares the given list of TX/RXs for each of the uploaded TX/RX sequences. Parameters: see setSubsequences.
     *
     * In contrast to setSubsequences, this method does not require the device to be stopped: the new sub-sequence
     * is programmed in the part of the sequencer memory that is currently not executed (sequencer double-buffering),
     * while the device can still acquire data with the current sub-sequence.
     *
     * When the device is running, the new sub-sequence is executed starting from the SECOND call to trigger after
     * this method: while waiting for a trigger, the sequencer has already moved to the first TX/RX of the next
     * acquisition, so the next trigger still acquires the data with the current sub-sequence.
     *
     * When the device is running, the following is required:
     * - the MANUAL work mode,
     * - the new sub-sequence produces data with exactly the same layout as the current one (e.g. the same number
     *   of TX/RXs), i.e. the output buffer is kept,
     * - the number of the host buffer elements is equal to the RX buffer size.
     * When the device is stopped, this method is equivalent to setSubsequences.
     *
     * @return the buffer and the metadata that describe the data acquired starting from the next trigger
     */
    virtual std::pair<std::shared_ptr<framework::Buffer>, std::vector<std::shared_ptr<session::Metadata>>>
    prepareSubsequences(const std::vector<std::vector<uint16>> &ops, const std::vector<std::optional<float>> &sris) = 0;

    Ultrasound(Ultrasound const &) = delete;
    Ultrasound(Ultrasound const &&) = delete;
    void operator=(Ultrasound const &) = delete;
    void operator=(Ultrasound const &&) = delete;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_ULTRASOUND_H
