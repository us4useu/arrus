#ifndef ARRUS_CORE_API_DEVICES_ULTRASOUND_H
#define ARRUS_CORE_API_DEVICES_ULTRASOUND_H

#include <memory>


#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
// TODO(pjarosik) avoid using us4r specific objects here
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"

namespace arrus::devices {

/**
 * An interface to the ultrasound device.
 */
class Ultrasound: public Device {
public:
    using Handle = std::unique_ptr<Ultrasound>;

    explicit Ultrasound(const DeviceId &id) : Device(id) {}

    virtual ~Ultrasound() = default;

    virtual std::pair<
        std::shared_ptr<arrus::framework::Buffer>,
        std::shared_ptr<arrus::session::Metadata>
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
     * Returns the current PGA gain value.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual uint16 getPgaGain() = 0;

    /**
     * Sets LNA gain.
     *
     * See docs of arrus::devices::RxSettings for more information.
     */
    virtual void setLnaGain(uint16 value) = 0;

    /**
     * Returns the current LNA gain value.
     *
     * See docs of arrus::devices::RxSettings for more information.
    */
    virtual uint16 getLnaGain() = 0;

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

    virtual void start() = 0;
    virtual void stop() = 0;

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
     * Enables High-Pass Filter and sets a given corner frequency.
     *
     * @param frequency corner high-pass filter frequency to set
     */
    virtual void setHpfCornerFrequency(uint32_t frequency) = 0;

    /**
     * Disables digital high-pass filter.
     */
    virtual void disableHpf() = 0;

    Ultrasound(Ultrasound const &) = delete;
    Ultrasound(Ultrasound const &&) = delete;
    void operator=(Ultrasound const &) = delete;
    void operator=(Ultrasound const &&) = delete;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_ULTRASOUND_H
