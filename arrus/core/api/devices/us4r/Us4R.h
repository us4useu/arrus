#ifndef ARRUS_CORE_DEVICES_US4R_US4R_H
#define ARRUS_CORE_DEVICES_US4R_US4R_H

#include <memory>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "FrameChannelMapping.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"

namespace arrus::devices {

/**
 * Us4R system: a group of Us4OEM modules and related components.
 *
 * By default system starts with IQ demodulator turned off.
 */
class Us4R : public DeviceWithComponents {
public:
    using Handle = std::unique_ptr<Us4R>;
    static constexpr long long INF_TIMEOUT = -1;

    explicit Us4R(const DeviceId &id) : DeviceWithComponents(id) {}

    ~Us4R() override = default;

    /**
     * Returns a handle to Us4OEM identified by given ordinal number.
     *
     * @param ordinal ordinal number of the us4oem to get
     * @return a handle to the us4oem module
     */
    virtual Us4OEM *getUs4OEM(Ordinal ordinal) = 0;

    /**
     * Returns a handle to an adapter identified by given ordinal number.
     *
     * @param ordinal ordinal number of the adapter to get
     * @return a handle to the adapter device
     */
    virtual ProbeAdapter::RawHandle getProbeAdapter(Ordinal ordinal) = 0;

    /**
     * Returns a handle to a probe identified by given ordinal number.
     *
     * @param ordinal ordinal number of the probe to get
     * @return a handle to the probe
     */
    virtual arrus::devices::Probe *getProbe(Ordinal ordinal) = 0;

    virtual std::pair<
        std::shared_ptr<arrus::framework::Buffer>,
        std::shared_ptr<arrus::devices::FrameChannelMapping>
    >
    upload(const ::arrus::ops::us4r::Scheme &scheme) = 0;

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
     * Returns measured HV voltage (plus).
     *
     * @return hv voltage measured by device [V]
     */
    virtual float getMeasuredPVoltage() = 0;

    /**
     * Returns measured HV voltage (minus).
     *
     * @return hv voltage measured by devivce [V]
     */
    virtual float getMeasuredMVoltage() = 0;

    /**
     * Disables HV voltage.
     */
    virtual void disableHV() = 0;

    /**
     * Equivalent to setTgcCurve(curve, true).
     */
    virtual void setTgcCurve(const std::vector<float> &tgcCurvePoints) = 0;

    /**
     * Sets TGC curve points asynchronously.
     *
     * Setting empty vector turns off analog TGC.
     * Setting non-empty vector turns off DTGC and turns on analog TGC.
     *
     * TGC curve can have up to 1022 samples.
     *
     * @param tgcCurvePoints tgc curve points to set.
     * @param applyCharacteristic set it to true if you want to compensate response characteristic (pre-computed
     * by us4us). If true, LNA and PGA gains should be set to 24 an 30 dB, respectively, otherwise an
     * ::arrus::IllegalArgumentException will be thrown.
     */
    virtual void setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) = 0;

    /**
     * Sets TGC curve points asynchronously.
     *
     * Setting empty vectors t and y turns off analog TGC. Setting non-empty vector turns off DTGC
     * and turns on analog TGC.
     *
     * Vectors t and y should have exactly the same size. The input t and y values will be interpolated
     * into target hardware sampling points (according to getCurrentSamplingFrequency and getCurrentTgcPoints).
     * Linear interpolation will be performed, the TGC curve will be extrapolated with the first (left-side of the cure)
     * and the last sample (right side of the curve).
     *
     * NOTE: TGC curve can have up to 1022 samples.
     *
     * @param t sampling time, relative to the "sample 0"
     * @param y values to apply at given sampling time
     * @param applyCharacteristic set it to true if you want to compensate response characteristic (pre-computed
     * by us4us). If true, LNA and PGA gains should be set to 24 an 30 dB, respectively, otherwise an
     * ::arrus::IllegalArgumentException will be thrown.
     */
    virtual void setTgcCurve(const std::vector<float> &t, const std::vector<float> &y, bool applyCharacteristic) = 0;

    /**
     * Returns us4R TGC sampling points (along time axis, relative to the "sample 0"), up to given maximum time.
     *
     * @param maxT maximum time range
     * @return TGC time points at which TGC curve sample takes place
     */
    virtual std::vector<float> getTgcCurvePoints(float maxT) const = 0;

    /**
     * Sets PGA gain.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setPgaGain(uint16 value) = 0;

    /**
     * Sets LNA gain.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setLnaGain(uint16 value) = 0;

    /**
     * Sets LPF cutoff.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setLpfCutoff(uint32 value) = 0;

    /**
     * Sets DTGC attenuation.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setDtgcAttenuation(std::optional<uint16> value) = 0;

    /**
    * Sets active termination.
    *
    * See docs of arrus::devices::RxSettings for more information.
    */
    virtual void setActiveTermination(std::optional<uint16> value) = 0;

    /**
    * Enables AFE auto offset removal.
    *
    */
    virtual void enableAfeAutoOffsetRemoval(void) = 0;

    /**
    * Disables AFE auto offset removal.
    *
    */
    virtual void disableAfeAutoOffsetRemoval(void) = 0;

    /**
    * Sets AFE auto offset removal accumulator cycles number to calculate average offset from.
	 * Possible settings are:
	 * 0 = 2047 cycles (default)
	 * 1 = 127 cycles
	 * 2 = 255 cycles
	 * 3 = 511 cycles
	 * 4 = 1023 cycles
	 * 5 = 2047 cycles
	 * 6 = 4095 cycles
	 * 7 = 8191 cycles
	 * 8 = 16383 cycles
	 * 9 = 32767 cycles
	 * 10 to 15 = 65535 cycles
	 *
    */
    virtual void setAfeAutoOffsetRemovalCycles(uint16_t cycles) = 0;

    /**
    * Sets AFE auto offset removal delay in reference clock cycles number from TX_TRIG.
    *
    */
    virtual void setAfeAutoOffsetRemovalDelay(uint16_t delay) = 0;

    /**
     * Sets a complete list of RxSettings on all Us4R components.
     *
     * @param settings settings to apply
     */
    virtual void setRxSettings(const RxSettings &settings) = 0;

    /**
     * If active is true, turns off probe's RX data acquisition and turns on test patterns generation.
     * Otherwise turns off test patterns generation and turns on probe's RX data acquisition.
     */
    virtual void setTestPattern(Us4OEM::RxTestPattern pattern) = 0;

    virtual void start() = 0;
    virtual void stop() = 0;

    virtual std::vector<unsigned short> getChannelsMask() = 0;

    /**
     * Returns the number of us4OEM modules that are used in this us4R system.
     */
    virtual uint8_t getNumberOfUs4OEMs() = 0;

    /**
     * Returns NOMINAL us4R device sampling frequency.
     */
    virtual float getSamplingFrequency() const = 0;


    /**
     * Returns the sampling frequency with which data from us4R will be acquired. The returned value
     * depends on the result of sequence upload (e.g. DDC decimation factor).
     */
    virtual float getCurrentSamplingFrequency() const = 0;

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

    Us4R(Us4R const &) = delete;
    Us4R(Us4R const &&) = delete;
    void operator=(Us4R const &) = delete;
    void operator=(Us4R const &&) = delete;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4R_H
