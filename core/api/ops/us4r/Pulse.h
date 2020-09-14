#ifndef ARRUS_CORE_API_OPS_US4R_PULSE_H
#define ARRUS_CORE_API_OPS_US4R_PULSE_H

namespace arrus::ops::us4r {

class Pulse {

public:
    Pulse(float centerFrequency, float nPeriods, bool inverse) :
        centerFrequency(centerFrequency), nPeriods(nPeriods),
        inverse(inverse) {}

    [[nodiscard]] float getCenterFrequency() const {
        return centerFrequency;
    }

    [[nodiscard]] float getNPeriods() const {
        return nPeriods;
    }

    [[nodiscard]] bool isInverse() const {
        return inverse;
    }

private:
    float centerFrequency;
    float nPeriods;
    bool inverse;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_PULSE_H
