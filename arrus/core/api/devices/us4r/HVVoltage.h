#ifndef ARRUS_CORE_API_DEVICES_US4R_HVVOLTAGE_H
#define ARRUS_CORE_API_DEVICES_US4R_HVVOLTAGE_H

namespace arrus::devices {
/**
* HV voltage curve description.
* Can be specified only by voltage negative and positive amplitude values.
*/
class HVVoltage {
public:
    /**
     * HV Voltage constructor.
     *
     * @param voltageMinus negative voltage to set [V]
     * @param voltagePlus  positive voltage to set [V]
     */
    HVVoltage(const Voltage voltageMinus, const Voltage voltagePlus)
        : voltageMinus(voltageMinus), voltagePlus(voltagePlus) {}

    HVVoltage(): voltageMinus(static_cast<Voltage>(0)), voltagePlus(static_cast<Voltage>(0)) {}

    Voltage getVoltageMinus() const { return voltageMinus; }
    Voltage getVoltagePlus() const { return voltagePlus; }

private:
    Voltage voltageMinus;
    Voltage voltagePlus;
};
}// namespace arrus::devices

#endif// ARRUS_CORE_API_DEVICES_US4R_HVVOLTAGE_H