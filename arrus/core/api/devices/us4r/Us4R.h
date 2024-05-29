#ifndef ARRUS_CORE_DEVICES_US4R_US4R_H
#define ARRUS_CORE_DEVICES_US4R_US4R_H

#include <memory>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "FrameChannelMapping.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/Ultrasound.h"
#include "arrus/core/api/session/Metadata.h"

namespace arrus::devices {

/**
 * Us4R system: a group of Us4OEM modules and related components.
 *
 * By default system starts with IQ demodulator turned off.
 */
class Us4R: public Ultrasound, public DeviceWithComponents {
public:
    using Handle = std::unique_ptr<Us4R>;
    static constexpr long long INF_TIMEOUT = -1;

    explicit Us4R(const DeviceId &id) : Ultrasound(id), DeviceWithComponents() {}
    ~Us4R() override = default;
    using Device::getDeviceId;

    /**
     * Returns a handle to Us4OEM identified by given ordinal number.
     *
     * @param ordinal ordinal number of the us4oem to get
     * @return a handle to the us4oem module
     */
    virtual Us4OEM *getUs4OEM(Ordinal ordinal) = 0;

    std::pair<framework::Buffer::SharedHandle, std::vector<session::Metadata::SharedHandle>>
    upload(const ::arrus::ops::us4r::Scheme &scheme) override = 0;

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
    virtual unsigned char getVoltage()  = 0;

    /**
     * Returns measured HV voltage (plus).
     *
     * @return hv voltage measured by device [V]
     */
    virtual float getMeasuredPVoltage()  = 0;

    /**
     * Returns measured HV voltage (minus).
     *
     * @return hv voltage measured by devivce [V]
     */
    virtual float getMeasuredMVoltage()  = 0;

    /**
     * Gets positive HV voltage measurement by UCD chip on OEM.
     *
     * @param oemId OEM ID
     * @return positive HV voltage UCD measurement [V]
     */
    virtual float getUCDMeasuredHVPVoltage(uint8_t oemId) = 0;

    /**
     * Gets negative HV voltage measurement by UCD chip on OEM.
     *
     * @param oemId OEM ID
     * @return negative HV voltage UCD measurement [V]
     */
    virtual float getUCDMeasuredHVMVoltage(uint8_t oemId) = 0;

    /**
     * Disables HV voltage.
     */
    virtual void disableHV()  = 0;

    /**
     * Equivalent to setTgcCurve(curve, true).
     */
    virtual void setTgcCurve(const std::vector<float> &tgcCurvePoints)  = 0;

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
    virtual void setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic)  = 0;

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
    virtual void setTgcCurve(const std::vector<float> &t, const std::vector<float> &y, bool applyCharacteristic)  = 0;

    /**
     * Returns us4R TGC sampling points (along time axis, relative to the "sample 0"), up to given maximum time.
     *
     * @param maxT maximum time range
     * @return TGC time points at which TGC curve sample takes place
     */
    virtual std::vector<float> getTgcCurvePoints(float maxT) const  = 0;

    /**
     * Sets PGA gain.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setPgaGain(uint16 value)  = 0;

    /**
     * Returns the current PGA gain value.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual uint16 getPgaGain()  = 0;

    /**
     * Sets LNA gain.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setLnaGain(uint16 value)  = 0;

    /**
     * Returns the current LNA gain value.
     *
     * See docs of arrus::devices::RxSettings for more information.
    */
    virtual uint16 getLnaGain()  = 0;

    /**
     * Sets LPF cutoff.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setLpfCutoff(uint32 value)  = 0;

    /**
     * Sets DTGC attenuation.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setDtgcAttenuation(std::optional<uint16> value)  = 0;

    /**
    * Sets active termination.
    *
    * See docs of arrus::devices::RxSettings for more information.
    */
    virtual void setActiveTermination(std::optional<uint16> value)  = 0;

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

    virtual void start() override = 0;
    virtual void stop() override = 0;
    virtual void trigger(bool sync, std::optional<long long> timeout) override = 0;

    /**
     * Synchronization point with us4R system. After returning from this method, the last "TX/RX" (triggered by the
     * trigger method will be  fully executed by the system.
     *
     * Sync with "SEQ_IRQ" interrupt (i.e. wait until the SEQ IRQ will occur).
     *
     * @param timeout timeout in number of milliseconds
     */
    virtual void sync(std::optional<long long> timeout) override = 0;

    virtual std::vector<unsigned short> getChannelsMask(Ordinal probeNumber) = 0;

    /**
     * Returns the number of us4OEM modules that are used in this us4R system.
     */
    virtual uint8_t getNumberOfUs4OEMs() = 0;

    /**
     * Returns the number of probes that are connected to the system.
     */
    int getNumberOfProbes() const = 0;

    /**
     * Returns NOMINAL us4R device sampling frequency.
     */
    virtual float getSamplingFrequency() const override = 0;


    /**
     * Returns the sampling frequency with which data from us4R will be acquired. The returned value
     * depends on the result of sequence upload (e.g. DDC decimation factor).
     */
    virtual float getCurrentSamplingFrequency() const override = 0;

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

    /**
     * Enables High-Pass Filter and sets a given corner frequency.
     *
     * Available corner frequency values (Hz): 4520'000, 2420'000, 1200'000, 600'000, 300'000, 180'000,
     * 80'000, 40'000, 20'000.
     *
     * @param frequency corner high-pass filter frequency to set
     */
    virtual void setHpfCornerFrequency(uint32_t frequency)  = 0;

    /**
     * Reads AFE register
     *
     * @param reg register address
     */
    virtual uint16_t getAfe(uint8_t reg) = 0;

    /**
     * Writes AFE register
     *
     * @param reg register address
     * @param val register value
     */
    virtual void setAfe(uint8_t reg, uint16_t val) = 0;


    /**
     * Disables digital high-pass filter.
     */
    virtual void disableHpf()  = 0;

    /**
     * Returns serial number of the backplane (if available).
     */
    virtual const char* getBackplaneSerialNumber() = 0;

    /**
     * Returns serial number of the backplane (if available).
     */
    virtual const char *getBackplaneRevision() = 0;

    std::pair<std::shared_ptr<framework::Buffer>, std::shared_ptr<session::Metadata>>
    setSubsequence(uint16 start, uint16 end, const std::optional<float> &sri) override = 0;

    virtual void setIOBitstream(unsigned short id, const std::vector<unsigned char> &levels, const std::vector<unsigned short> &periods) = 0;

    /**
     * Returns probe identified by given ordinal number.
     *
     * @param ordinal ordinal number of the probe to get
     * @return probe handle
     */
    Probe* getProbe(Ordinal ordinal) override = 0;

    /**
     * Sets maximum pulse length that can be set during the TX/RX sequence programming.
     * std::nullopt means to use up to 32 TX cycles.
     *
     * @param maxLength maxium pulse length (s) nullopt means to use 32 TX cycles (legacy OEM constraint)
     */
    virtual void setMaximumPulseLength(std::optional<float> maxLength) = 0;

    Us4R(Us4R const &) = delete;
    Us4R(Us4R const &&) = delete;
    void operator=(Us4R const &) = delete;
    void operator=(Us4R const &&) = delete;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4R_H
