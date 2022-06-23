#include <utility>

#include "arrus/core/api/ops/us4r/DigitalDownConversion.h"
#include "arrus/core/common/collections.h"

namespace arrus::ops::us4r {

class DigitalDownConversion::Impl {
public:
    Impl(float demodulationFrequency, float decimationFactor, std::vector<float> firCoefficients)
        : demodulationFrequency(demodulationFrequency), decimationFactor(decimationFactor),
          firCoefficients(std::move(firCoefficients)) {}

    inline float getDemodulationFrequency() const { return demodulationFrequency; }
    inline float getDecimationFactor() const { return decimationFactor; }
    inline const Span<float> getFirCoefficients() const { return firCoefficients; }

private:
    float demodulationFrequency, decimationFactor;
    std::vector<float> firCoefficients;
};

DigitalDownConversion::DigitalDownConversion(float demodulationFrequency, Span<float> firCoefficients,
                                             float decimationFactor) {
    this->impl = UniqueHandle<Impl>::create(demodulationFrequency, decimationFactor, copyToVector(firCoefficients));
}

float DigitalDownConversion::getDemodulationFrequency() const { return impl->getDemodulationFrequency(); }
Span<float> DigitalDownConversion::getFirCoefficients() const { return impl->getFirCoefficients(); }
float DigitalDownConversion::getDecimationFactor() const { return impl->getDecimationFactor(); }

DigitalDownConversion::DigitalDownConversion(const DigitalDownConversion &o) = default;
DigitalDownConversion::DigitalDownConversion(DigitalDownConversion &&o)  noexcept = default;
DigitalDownConversion::~DigitalDownConversion() {}
DigitalDownConversion &DigitalDownConversion::operator=(const DigitalDownConversion &o) = default;
DigitalDownConversion &DigitalDownConversion::operator=(DigitalDownConversion &&o) noexcept = default;

}// namespace arrus::ops::us4r