#ifndef ARRUS_CORE_API_OPS_US4R_PULSE_H
#define ARRUS_CORE_API_OPS_US4R_PULSE_H

namespace arrus::ops::us4r {

/**
 * A single pulse (sine wave) produced by us4r device.
 */
class Pulse {

public:
	/**
	 * Pulse constructor.
	 *
	 * @param centerFrequency center frequency of the transmitted pulse
	 * @param nPeriods pulse number of periods, should be a multiple of 0.5
	 * @param inverse if set to true - inverse the pulse polarity
	 */
    Pulse(float centerFrequency, float nPeriods, bool inverse):
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
