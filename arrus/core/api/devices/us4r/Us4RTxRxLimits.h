#ifndef ARRUS_CORE_API_DEVICES_US4R_US4RTXRXLIMITS_H
#define ARRUS_CORE_API_DEVICES_US4R_US4RTXRXLIMITS_H

#include <optional>

#include "arrus/core/api/common/Interval.h"
#include "std4us/Optional.hpp"
#include "std4us/StdInterop.hpp"

namespace arrus::devices {

/**
 * Custom TX/RX limits to be applied on the TX/RX sequence validation.
 * NOTE: all the values are optional; nullopt means that the default value for a given
 * us4OEM revision will be used.
 *
 * Storage uses std4us::Optional for ABI stability. std::optional accessors
 * remain as inline header-only backward-compatibility shims.
 */
class Us4RTxRxLimits {
public:
    Us4RTxRxLimits(const std::optional<Interval<float>> &pulseLength,
                   const std::optional<Interval<Voltage>> &voltage,
                   const std::optional<Interval<float>> &pri)
        : pulseLength(std4us::fromStd(pulseLength)),
          pri(std4us::fromStd(pri)),
          voltage(std4us::fromStd(voltage)) {}

    Us4RTxRxLimits(std4us::Optional<Interval<float>> pulseLength,
                   std4us::Optional<Interval<Voltage>> voltage,
                   std4us::Optional<Interval<float>> pri)
        : pulseLength(std::move(pulseLength)), pri(std::move(pri)), voltage(std::move(voltage)) {}

    std::optional<Interval<float>> getPulseLength() const { return std4us::toStd(pulseLength); }
    std::optional<Interval<float>> getPri() const { return std4us::toStd(pri); }
    std::optional<Interval<Voltage>> getVoltage() const { return std4us::toStd(voltage); }

    const std4us::Optional<Interval<float>> &getPulseLengthNative() const { return pulseLength; }
    const std4us::Optional<Interval<float>> &getPriNative() const { return pri; }
    const std4us::Optional<Interval<Voltage>> &getVoltageNative() const { return voltage; }

private:
    std4us::Optional<Interval<float>> pulseLength;
    std4us::Optional<Interval<float>> pri;
    std4us::Optional<Interval<Voltage>> voltage;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_US4R_US4RTXRXLIMITS_H
