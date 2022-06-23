#ifndef ARRUS_CORE_API_OPS_US4R_DIGITALDOWNCONVERSION_H
#define ARRUS_CORE_API_OPS_US4R_DIGITALDOWNCONVERSION_H

#include <vector>

#include "arrus/core/api/common.h"

namespace arrus::ops::us4r {

class ARRUS_CPP_EXPORT DigitalDownConversion {
    class Impl;
    UniqueHandle<Impl> impl;
public:

    DigitalDownConversion(float demodulationFrequency, Span<float> firCoefficients, float decimationFactor);
    DigitalDownConversion(const DigitalDownConversion &o);
    DigitalDownConversion(DigitalDownConversion &&o) noexcept;
    virtual ~DigitalDownConversion();
    DigitalDownConversion& operator=(const DigitalDownConversion &o);
    DigitalDownConversion& operator=(DigitalDownConversion &&o) noexcept;

    float getDemodulationFrequency() const;
    Span<float> getFirCoefficients() const;
    float getDecimationFactor() const;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_DIGITALDOWNCONVERSION_H
