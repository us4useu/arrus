#ifndef ARRUS_CORE_API_OPS_US4R_PULSE_H
#define ARRUS_CORE_API_OPS_US4R_PULSE_H

namespace arrus::ops::us4r {

class Pulse {

public:
    Pulse(float centerFrequency, float nPeriods, bool inverse) :
        centerFrequency(centerFrequency), nPeriods(nPeriods),
        inverse(inverse) {}

    float getCenterFrequency() const {
        return centerFrequency;
    }

    float getNPeriods() const {
        return nPeriods;
    }

    bool isInverse() const {
        return inverse;
    }

    bool operator==(const Pulse &rhs) const {
        return centerFrequency == rhs.centerFrequency
               && nPeriods == rhs.nPeriods
               && inverse == rhs.inverse;
    }

    bool operator!=(const Pulse &rhs) const {
        return !(rhs == *this);
    }

private:
    float centerFrequency;
    float nPeriods;
    bool inverse;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_PULSE_H
