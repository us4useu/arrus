#ifndef ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXLIMITS_H
#define ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXLIMITS_H

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/Interval.h"

namespace arrus::ops::us4r {

class TxLimitsBuilder;

/**
 * RX op limits.
 *
 * The instance of this class defines what constraints are applied on the RX op parameters.
 */
class TxLimits {
public:
    TxLimits(const Interval<float> &frequency, const Interval<float> &delay, const Interval<float> &pulseCycles,
             const Interval<Voltage> &voltage)
        : frequency(frequency), delay(delay), pulseCycles(pulseCycles), voltage(voltage) {}

    const Interval<float> &getFrequency() const { return frequency; }
    const Interval<float> &getPulseLength() const { return pulseLength; }
    const Interval<float> &getPulseCycles() const { return pulseCycles; }
    const Interval<Voltage> &getVoltage() const { return voltage; }
    const Interval<float> &getDelay() const { return delay; }

private:
    friend class TxLimitsBuilder;
    Interval<float> frequency;
    Interval<float> delay;
    Interval<float> pulseLength;
    Interval<float> pulseCycles;
    Interval<Voltage> voltage;
};

class TxLimitsBuilder {

public:
    explicit TxLimitsBuilder(const TxLimits tx) : tx(std::move(tx)) {}

    TxLimitsBuilder& setFrequency(const Interval<float> &frequency) { tx->frequency = frequency; return *this;  }
    TxLimitsBuilder& setDelay(const Interval<float> &delay) { tx->delay = delay; return *this; }
    TxLimitsBuilder& setPulseLength(const Interval<float> &pulseLength) { tx->pulseLength = pulseLength; return *this; }
    TxLimitsBuilder& setPulseCycles(const Interval<float> &pulseCycles) { tx->pulseCycles = pulseCycles; return *this; }
    TxLimitsBuilder& setVoltage(const Interval<Voltage> &voltage) { tx->voltage = voltage; return *this; }

    TxLimits build() {
        return tx.value();
    }

private:
    std::optional<TxLimits> tx;
};


}


#endif //ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXLIMITS_H
