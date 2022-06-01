#ifndef ARRUS_CORE_API_DEVICES_ULTRASOUND_H
#define ARRUS_CORE_API_DEVICES_ULTRASOUND_H

#include <memory>
#include <vector>

namespace arrus::devices {

class Ultrasound {

public:
    using Handle = std::unique_ptr<Ultrasound>;

    ~Ultrasound() override = default;

    /**
     * Returns a handle to a probe identified by given ordinal number.
     *
     * @param ordinal ordinal number of the probe to get
     * @return a handle to the probe
     */
    virtual arrus::devices::Probe* getProbe(Ordinal ordinal) = 0;

    virtual std::pair<
    std::shared_ptr<arrus::framework::Buffer>,
    std::shared_ptr<arrus::devices::FrameChannelMapping>
    >
    upload(const ::arrus::ops::us4r::TxRxSequence &seq, unsigned short rxBufferSize,
           const ::arrus::ops::us4r::Scheme::WorkMode &workMode,
           const ::arrus::framework::DataBufferSpec &hostBufferSpec) = 0;

    /**
     * Sets HV voltage.
     *
     * @param voltage voltage to set [V]
     */
    virtual void setVoltage(Voltage voltage) = 0;

    /**
     * Returns configured HV voltage.
     *
     * @return hv voltage value configured on device [V]
     */
    virtual unsigned char getVoltage() = 0;

    /**
     * Equivalent to setTgcCurve(curve, true).
     */
    virtual void setTgcCurve(const std::vector<float>& tgcCurvePoints) = 0;

    /**
     * Sets a complete list of RxSettings on all Us4R components.
     *
     * @param settings settings to apply
     */
    virtual void setRxSettings(const RxSettings &settings) = 0;

    virtual void start() = 0;
    virtual void stop() = 0;

    virtual std::vector<unsigned short> getChannelsMask() = 0;

    /**
     * Returns the number of us4OEM modules that are used in this us4R system.
     */
    virtual uint8_t getNumberOfUs4OEMs() = 0;

    /**
     * Returns us4R device sampling frequency.
     */
    virtual float getSamplingFrequency() const = 0;

    /**
     * Checks state of the Us4R device. Currently checks if each us4OEM module is in
     * the correct state.
     *
     * @throws arrus::IllegalStateException when some inconsistent state was detected
     */
    virtual void checkState() const = 0;

    /**
     * Set the system to stop when (RX or host) buffer overflow is detected.
     *
     * This property is set by default to true.
     *
     * @param isStopOnOverflow whether the system should stop when buffer overflow is detected.
     */
    virtual void setStopOnOverflow(bool isStopOnOverflow) = 0;

    /**
     * Returns true if the system will be stopped when (RX of host) buffer overflow is detected.
     *
     * This property is set by default to true.
     *
     * @param isStopOnOverflow whether the system should stop when buffer overflow is detected.
     */
    virtual bool isStopOnOverflow() const = 0;

    Ultrasound(Ultrasound const&) = delete;
    Ultrasound(Ultrasound const&&) = delete;
    void operator=(Ultrasound const&) = delete;
    void operator=(Ultrasound const&&) = delete;


private:
    UltrasoundImpl impl;
};

}

#endif//ARRUS_CORE_API_DEVICES_ULTRASOUND_H
