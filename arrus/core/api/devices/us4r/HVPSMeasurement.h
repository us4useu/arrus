#ifndef ARRUS_CORE_API_DEVICES_US4R_HVPS_MEASUREMENT_H
#define ARRUS_CORE_API_DEVICES_US4R_HVPS_MEASUREMENT_H
#include "HVVoltage.h"
#include "arrus/core/api/ops/us4r/Pulse.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace arrus::devices {

class HVPSMeasurementBuilder;

/**
 * HVPS time-based measurement.
 *
 * NOTE: this class assumes us4OEM HV rail numbering instead of amplitude level numbering
 * (mapping: rail 0 -> level 2, rail 1 -> level 1).
 */
class HVPSMeasurement {
public:
    enum Unit { VOLTAGE, CURRENT };
    enum Polarity { MINUS, PLUS };
    using AmplitudeLevel = uint8;

    const std::vector<float> &get(AmplitudeLevel rail, Polarity polarity, Unit unit) const {
        return measurements.at(rail).at(polarity).at(unit);
    }
private:
    explicit HVPSMeasurement(const std::vector<std::vector<std::vector<std::vector<float>>>> &measurements)
        : measurements(measurements) {}
    friend HVPSMeasurementBuilder;
    // rail, polarity, unit, time -> value
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

    void set(AmplitudeLevel rail, Polarity polarity, Unit unit, std::vector<float> measurement) {
        measurements[rail][polarity][unit] = std::move(measurement);
    }

    HVPSMeasurement build() { return HVPSMeasurement{measurements}; }

private:
    std::vector<std::vector<std::vector<std::vector<float>>>> measurements;
};

class HVPSScalarMeasurementBuilder;

/**
 * HVPS scalar measurement (from a single point in time).
 *
 * NOTE: this class assumes us4OEM HV rail numbering instead of amplitude level numbering
 * (mapping: rail 0 -> level 2, rail 1 -> level 1).
 */
class HVPSScalarMeasurement {
public:
    enum Polarity { MINUS, PLUS };
    using AmplitudeLevel = uint8;

    float getVoltage(AmplitudeLevel rail, Polarity polarity) const {
        return measurements.at(rail).at(polarity);
    }
private:
    explicit HVPSScalarMeasurement(const std::vector<std::vector<float>> &measurements)
        : measurements(measurements) {}
    friend HVPSScalarMeasurementBuilder;
    // rail (0, 1), polarity -> value
    std::vector<std::vector<float>> measurements;
};

class HVPSScalarMeasurementBuilder {
public:
    using AmplitudeLevel = HVPSScalarMeasurement::AmplitudeLevel;
    using Polarity = HVPSScalarMeasurement::Polarity;

    explicit HVPSScalarMeasurementBuilder() {
        measurements = std::vector{2, std::vector{2, 0.0f}};
    }

    void set(AmplitudeLevel rail, Polarity polarity, float value) {
        measurements.at(rail).at(polarity) = value;
    }

    HVPSScalarMeasurement build() { return HVPSScalarMeasurement{measurements}; }

private:
    std::vector<std::vector<float>> measurements;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_US4R_HVPS_MEASUREMENT_H
