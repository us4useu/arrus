#ifndef ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXLIMITS_H
#define ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXLIMITS_H

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/Interval.h"

namespace arrus::ops::us4r {

/**
 * RX op limits.
 *
 * The instance of this class defines what constraints are applied on the RX op parameters.
 */
class TxLimits {
public:
    TxLimits(const Interval<float> &frequency, const Interval<float> &delay, const Interval<float> &pulseLength,
             const Interval<Voltage> &voltage)
        : frequency(frequency), delay(delay), pulseLength(pulseLength), voltage(voltage) {}

    const Interval<float> &getFrequency() const { return frequency; }
    const Interval<float> &getPulseLength() const { return pulseLength; }
    const Interval<Voltage> &getVoltage() const { return voltage; }
    const Interval<float> &getDelay() const { return delay; }

private:
    Interval<float> frequency;
    Interval<float> delay;
    Interval<float> pulseLength;
    Interval<Voltage> voltage;
};


}


#endif //ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXLIMITS_H
