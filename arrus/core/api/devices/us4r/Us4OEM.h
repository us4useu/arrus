#ifndef ARRUS_CORE_API_DEVICES_US4R_US4OEM_H
#define ARRUS_CORE_API_DEVICES_US4R_US4OEM_H

#include <memory>
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/TriggerGenerator.h"

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
    //virtual void setAfeFir(uint8_t address, uint16_t* coeffs, uint8_t length) = 0;

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
    * Sets AFE auto offset removal accumulator cycles number.
    *
    * @param cycles: number of cycles is given by N=(2^(cycles+6))-1
    */
    virtual void setAfeAutoOffsetRemovalCycles(uint8_t cycles) = 0;

    /**
    * Sets AFE auto offset removal delay from TX_TRIG
    *
    * @param delay: delay from TX_TRIG in reference clock cycles
    */
    virtual void setAfeAutoOffsetRemovalDelay(uint8_t delay) = 0;

    /**
    * Enables AFE high pass filter.
    *
    */
    virtual void enableAfeHPF(void) = 0;

    /**
    * Enables AFE high pass filter.
    *
    */
    virtual void disableAfeHPF(void) = 0;

    /**
    * Sets AFE high pass filter corner frequency.
    *
    /**
    * Sets AFE high pass filter corner frequency (default is 300 kHz, k=6)
    *
    * @param k sets HPF corner frequency
     * Possible settings (for 65 MHz reference clock) are:
     * 2 = 4520 kHz
     * 3 = 2420 kHz
     * 4 = 1200 kHz
     * 5 = 600 kHz
     * 6 = 300 kHz (default)
     * 7 = 180 kHz
     * 8 = 80 kHz
     * 9 = 40 kHz
     * 10 = 20 kHz
    */
    virtual void setAfeHPFCornerFrequency(uint8_t k) = 0;

    /**
    * Enables and configures AFE built-in demodulator
    * 
    * @param demodulationFrequency: Demodulation frequency
    * @param decimationFactor: Decimation factor
    * @param firCoefficients: Pointer to Low pass filter coefficients buffer
    * @param nCoefficients: Number of FIR coefficients
    * @throws arrus::IllegalStateException when invalid input parameters detected
    */
    virtual void setAfeDemod(float demodulationFrequency, float decimationFactor, const int16 *firCoefficients,
                             size_t nCoefficients) = 0;

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

    Us4OEM(Us4OEM const&) = delete;
    Us4OEM(Us4OEM const&&) = delete;
    void operator=(Us4OEM const&) = delete;
    void operator=(Us4OEM const&&) = delete;
protected:
    explicit Us4OEM(const DeviceId &id): Device(id) {}
};


}

#endif // ARRUS_CORE_API_DEVICES_US4R_US4OEM_H
