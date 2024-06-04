#ifndef ARRUS_CORE_API_DEVICES_US4R_US4RTXRXLIMITS_H
#define ARRUS_CORE_API_DEVICES_US4R_US4RTXRXLIMITS_H

#include <optional>
#include "arrus/core/api/common/Interval.h"

namespace arrus::devices {

/**
 * Custom TX/RX limits to be applied on the TX/RX sequence validation.
 * NOTE: all the values are optional; nullopt means that the default value for a given
 * us4OEM revision will be used.
 */
class Us4RTxRxLimits {
public:
    Us4RTxRxLimits(const std::optional<Interval<float>> &pulseLength,
                   const std::optional<Interval<Voltage>> &voltage,
                   const std::optional<Interval<float>> &pri)
        : pulseLength(pulseLength), pri(pri), voltage(voltage) {}

    const std::optional<Interval<float>> &getPulseLength() const { return pulseLength; }
    const std::optional<Interval<float>> &getPri() const { return pri; }
    const std::optional<Interval<Voltage>> &getVoltage() const { return voltage; }

private:
    std::optional<Interval<float>> pulseLength;
    std::optional<Interval<float>> pri;
    std::optional<Interval<Voltage>> voltage;
};

}

#endif//ARRUS_CORE_API_DEVICES_US4R_US4RTXRXLIMITS_H
