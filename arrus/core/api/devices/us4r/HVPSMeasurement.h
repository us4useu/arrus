#ifndef ARRUS_CORE_API_DEVICES_US4R_HVPS_MEASUREMENT_H
#define ARRUS_CORE_API_DEVICES_US4R_HVPS_MEASUREMENT_H
#include "arrus/core/api/ops/us4r/Pulse.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace arrus::devices {

class HVPSMeasurementBuilder;

class HVPSMeasurement {
public:
    enum Unit { VOLTAGE, CURRENT };
    enum Polarity { MINUS, PLUS };
    using AmplitudeLevel = uint8;

    const std::vector<float> &get(AmplitudeLevel level, Polarity polarity, Unit unit) const {
        return measurements.at(level).at(polarity).at(unit);
    }
private:
    explicit HVPSMeasurement(const std::vector<std::vector<std::vector<std::vector<float>>>> &measurements)
        : measurements(measurements) {}
    friend HVPSMeasurementBuilder;
    // level, polarity, unit, time -> value
    std::vector<std::vector<std::vector<std::vector<float>>>> measurements;
};

class HVPSMeasurementBuilder {
public:
    using AmplitudeLevel = HVPSMeasurement::AmplitudeLevel;
    using Polarity = HVPSMeasurement::Polarity;
    using Unit = HVPSMeasurement::Unit;

    explicit HVPSMeasurementBuilder() {
        measurements = std::vector{2, std::vector{2, std::vector{2, std::vector<float>{}}}};
    }

    void set(AmplitudeLevel level, Polarity polarity, Unit unit, std::vector<float> measurement) {
        measurements[level][polarity][unit] = std::move(measurement);
    }

    HVPSMeasurement build() { return HVPSMeasurement{measurements}; }

private:
    std::vector<std::vector<std::vector<std::vector<float>>>> measurements;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_US4R_HVPS_MEASUREMENT_H