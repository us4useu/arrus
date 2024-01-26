#ifndef ARRUS_CORE_API_DEVICES_US4R_US4OEM_H
#define ARRUS_CORE_API_DEVICES_US4R_US4OEM_H

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/TriggerGenerator.h"
#include "arrus/core/api/devices/us4r/HVPSMeasurement.h"
#include <memory>

namespace arrus::devices {

class Us4OEM : public Device, public TriggerGenerator {
public:
    using Handle = std::unique_ptr<Us4OEM>;
    using RawHandle = PtrHandle<Us4OEM>;

   /**
    * Us4OEM ADC test pattern state.
    */
    enum class RxTestPattern {
        OFF,
        /** Ramp (sawtooth data pattern). */
        RAMP,
    };

    ~Us4OEM() override = default;

    /**
     * Returns nominal sampling frequency on the us4OEM device.
     */
    virtual float getSamplingFrequency() = 0;

    /**
     * Returns current sampling frequency of the us4OEM device.
     */
    virtual float getCurrentSamplingFrequency() const = 0;

    /**
     * Returns temperature measured by Us4OEM's FPGA [Celsius].
     */
    virtual float getFPGATemperature() = 0;

    /**
     * Returns temperature measured by Us4OEM's UCD [Celsius]
     */
    virtual float getUCDTemperature() = 0;

    /**
     * Returns external temperature measured by Us4OEM's UCD [Celsius]
     */
    virtual float getUCDExternalTemperature() = 0;

    /**
     * Returns rail voltage measured by Us4OEM's UCD [V].
     *
     * @param rail UCD rail number
     */
    virtual float getUCDMeasuredVoltage(uint8_t rail) = 0;

    /**
    * Reads AFE register
    *
    * @param address: register address
    * @return: register value
    * @throws arrus::IllegalStateException when invalid input parameters detected
    */
	virtual uint16_t getAfe(uint8_t address) = 0;

    /**
    * Writes AFE register
    *
    * @param address: register address
    * @param value: register value
    * @throws arrus::IllegalStateException when invalid input parameters detected
    */
	virtual void setAfe(uint8_t address, uint16_t value) = 0;

    /**
    * Enables and configures AFE built-in demodulator
    *
    * @param demodulationFrequency: Demodulation frequency
    * @param decimationFactor: Decimation factor
    * @param firCoefficients: Pointer to Low pass filter coefficients buffer
    * @param nCoefficients: Number of FIR coefficients
    * @throws arrus::IllegalStateException when invalid input parameters detected
    */
    virtual void setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                             size_t nCoefficients) = 0;

    /**
    * Disables AFE built-in demodulator
    *
    */
    virtual void disableAfeDemod() = 0;

    /**
     * Checks if the firmware version on the Us4OEM module is correct.
     *
     * @throws ::arrus::IllegalStateException when the incorrect version was detected.
     */
    virtual void checkFirmwareVersion() = 0;

    /**
     * Checks if the us4OEM is in the correct state (as seen by host PC).
     *
     * Note: currently only the firmware version is checked (to verify if the us4OEM module
     * memory space is still available for the us4OEM module).
     *
     * @throws arrus::IllegalStateException when the incorrect version was detected.
     */
    virtual void checkState() = 0;

    /**
     * Returns firmware version installed on the us4OEM module.
     */
    virtual uint32 getFirmwareVersion() = 0;

    /**
     * Returns Tx component firmware version installed on this us4OEM module.
     */
    virtual uint32 getTxFirmwareVersion() = 0;

    /**
     * Returns Tx offset
    */
    virtual uint32_t getTxOffset() = 0;

    /**
     * Returns OEM version (OEM/OEM+)
    */
    virtual uint32_t getOemVersion() = 0;

    /**
     * Returns current FPGA wall clock (time passed since Init function was called).
     *
     * @return FPGA wall clock (seconds)
     */
    virtual float getFPGAWallclock() = 0;

    /**
     * Enables High-Pass Filter and sets a given corner frequency.
     *
     * Available corner frequency values (Hz): 4520'000, 2420'000, 1200'000, 600'000, 300'000, 180'000,
     * 80'000, 40'000, 20'000.
     *
     * @param frequency corner high-pass filter frequency to set
     */
    virtual void setHpfCornerFrequency(uint32_t frequency) = 0;

    /**
     * Returns serial number of this us4OEM (a null-terminated string).
     */
    virtual const char* getSerialNumber() = 0;

    /**
     * Returns revision number of this us4OEM (a null-terminated string).
     */
    virtual const char* getRevision() = 0;

    /**
     * Disables digital high-pass filter.
     */
    virtual void disableHpf() = 0;

    /**
     * Returns HVPS ADC measurements
     */
    virtual HVPSMeasurement getHVPSMeasurement() = 0;

    /**
     * Configures HVPS voltage/current measurement in Sync mode.
     *
     * :param nSamples: number of ADC samples to acquire.
     * :param frequency: Requested sampling frequency.
     * :return: Actual sampling frequency
     */
    virtual float setHVPSSyncMeasurement(uint16_t nSamples, float frequency) = 0;

    Us4OEM(Us4OEM const&) = delete;
    Us4OEM(Us4OEM const&&) = delete;
    void operator=(Us4OEM const&) = delete;
    void operator=(Us4OEM const&&) = delete;
protected:
    explicit Us4OEM(const DeviceId &id): Device(id) {}
};


}

#endif // ARRUS_CORE_API_DEVICES_US4R_US4OEM_H
