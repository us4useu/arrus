#ifndef ARRUS_CORE_API_OPS_US4R_DIGITALDOWNCONVERSION_H
#define ARRUS_CORE_API_OPS_US4R_DIGITALDOWNCONVERSION_H

#include <utility>
#include <vector>

#include "arrus/core/api/common.h"

namespace arrus::ops::us4r {

class ARRUS_CPP_EXPORT DigitalDownConversion {
    class Impl;
    UniqueHandle<Impl> impl;
public:
    /**
     * Us4R Digital Down Conversion block.
     *
     * Note: the decimation factor can also have a fractional part: 0.25, 0.5 or 0.75.
     *
     * Note: the FIR filter order (i.e. total number of taps)depends on the decimation factor
     * and should be equal: decimationFactor*16 for integer decimation factor; decimationFactor*32 for
     * decimation factor with fractional part 0.5; decimationFactor*64 for decimation facator with
     * fractional part 0.25 or 0.75.
     *
     * Note: only a upper half of the FIR filter coefficients should be provided.
     *
     * @param demodulationFrequency demodulation frequency to apply [Hz]
     * @param firCoefficients FIR filter coefficients to apply
     * @param decimationFactor decimation factor to apply, should be in range [2, 63]
     * @param gain an extra digital gain to apply (after decimation filter), by default set to 12 dB.
     *   Currently only 0 and 12 dB are supported [dB]
     */
    DigitalDownConversion(float demodulationFrequency, std::vector<float> firCoefficients, float decimationFactor,
                          float gain = 12.0)
    : DigitalDownConversion(demodulationFrequency, Span<float>{firCoefficients}, decimationFactor, gain) {}
    DigitalDownConversion(float demodulationFrequency, Span<float> firCoefficients, float decimationFactor,
                          float gain = 12.0);
    DigitalDownConversion(const DigitalDownConversion &o);
    DigitalDownConversion(DigitalDownConversion &&o) noexcept;
    virtual ~DigitalDownConversion();
    DigitalDownConversion& operator=(const DigitalDownConversion &o);
    DigitalDownConversion& operator=(DigitalDownConversion &&o) noexcept;

    float getDemodulationFrequency() const;
    Span<float> getFirCoefficients() const;
    float getDecimationFactor() const;
    /** Returns an extra digital gain to apply (after the decimation filter). */
    float getGain() const;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_DIGITALDOWNCONVERSION_H
