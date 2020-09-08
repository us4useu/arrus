#ifndef ARRUS_CORE_API_OPS_US4R_PULSE_H
#define ARRUS_CORE_API_OPS_US4R_PULSE_H

namespace arrus::ops::us4r {

class Pulse {

public:
    Pulse(double centerFrequency, double nPeriods, bool inverse) :
        centerFrequency(centerFrequency), nPeriods(nPeriods), inverse(inverse) {}

    [[nodiscard]] double getCenterFrequency() const {
        return centerFrequency;
    }

    [[nodiscard]] double getNPeriods() const {
        return nPeriods;
    }

    [[nodiscard]] bool isInverse() const {
        return inverse;
    }

private:
    double centerFrequency;
    double nPeriods;
    bool inverse;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_PULSE_H
