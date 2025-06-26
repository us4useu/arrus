#include <utility>

#include "arrus/core/api/ops/us4r/DigitalDownConversion.h"
#include "arrus/core/common/collections.h"

namespace arrus::ops::us4r {

class DigitalDownConversion::Impl {
public:
    Impl(float demodulationFrequency, float decimationFactor, std::vector<float> firCoefficients, float gain)
        : demodulationFrequency(demodulationFrequency), decimationFactor(decimationFactor),
          firCoefficients(std::move(firCoefficients)), gain(gain) {}

    inline float getDemodulationFrequency() const { return demodulationFrequency; }
    inline float getDecimationFactor() const { return decimationFactor; }
    inline const Span<float> getFirCoefficients() const {
        Span<float> s{firCoefficients.data(), firCoefficients.size()};
        return s;
    }
    inline float getGain() const { return gain; }

private:
    float demodulationFrequency, decimationFactor;
    std::vector<float> firCoefficients;
    float gain;
};

DigitalDownConversion::DigitalDownConversion(float demodulationFrequency, Span<float> firCoefficients,
                                             float decimationFactor, float gain) {
    this->impl = UniqueHandle<Impl>::create(demodulationFrequency, decimationFactor, copyToVector(firCoefficients),
                                            gain);
}

float DigitalDownConversion::getDemodulationFrequency() const { return impl->getDemodulationFrequency(); }
Span<float> DigitalDownConversion::getFirCoefficients() const { return impl->getFirCoefficients(); }
float DigitalDownConversion::getDecimationFactor() const { return impl->getDecimationFactor(); }
float DigitalDownConversion::getGain() const { return impl->getGain(); }

DigitalDownConversion::DigitalDownConversion(const DigitalDownConversion &o) = default;
DigitalDownConversion::DigitalDownConversion(DigitalDownConversion &&o)  noexcept = default;
DigitalDownConversion::~DigitalDownConversion() {}
DigitalDownConversion &DigitalDownConversion::operator=(const DigitalDownConversion &o) = default;
DigitalDownConversion &DigitalDownConversion::operator=(DigitalDownConversion &&o) noexcept = default;

}// namespace arrus::ops::us4r