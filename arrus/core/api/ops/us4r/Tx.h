#ifndef ARRUS_CORE_API_OPS_US4R_TX_H
#define ARRUS_CORE_API_OPS_US4R_TX_H

#include <utility>

#include "Pulse.h"
#include "Waveform.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::ops::us4r {

/**
 * A single pulse transmission.
 */
class Tx {
public:
    /**
	 * Tx constructor.
	 *
	 * @param aperture transmit aperture specified as a bit mask; aperture[i] means that the i-th channel should be turned on
	 * @param delays transmit delays to apply; delays[i] applies to channel i
	 * @param pulse pulse to transmit
	 * @parma placement probe on which the Tx should be performed
	 */
    Tx(std::vector<bool> aperture, std::vector<float> delays, const Pulse &pulse, devices::DeviceId placement)
        : aperture(std::move(aperture)), delays(std::move(delays)), excitation(pulse.toWaveform()), placement(placement) {
        if(placement.getDeviceType() != devices::DeviceType::Probe) {
            throw IllegalArgumentException("Only probe can be set as a placement for TX.");
        }
    }

    Tx(std::vector<bool> aperture, std::vector<float> delays, const Waveform &waveform, devices::DeviceId placement)
        : aperture(std::move(aperture)), delays(std::move(delays)), excitation(waveform), placement(placement) {
        if(placement.getDeviceType() != devices::DeviceType::Probe) {
            throw IllegalArgumentException("Only probe can be set as a placement for TX.");
        }
    }

    Tx(std::vector<bool> aperture, std::vector<float> delays, const Pulse &excitation)
        : aperture(std::move(aperture)), delays(std::move(delays)), excitation(excitation.toWaveform()),
          placement(devices::DeviceId(devices::DeviceType::Probe, 0)) {}

    Tx(std::vector<bool> aperture, std::vector<float> delays, const Waveform &waveform)
        : aperture(std::move(aperture)), delays(std::move(delays)), excitation(waveform),
          placement(devices::DeviceId(devices::DeviceType::Probe, 0)) {}

    const std::vector<bool> &getAperture() const { return aperture; }

    const std::vector<float> &getDelays() const { return delays; }

    const Waveform &getExcitation() const { return excitation; }

    const devices::DeviceId &getPlacement() const { return placement; }

    /**
     * Returns an array with delays for active (i.e. aperture[i] = true) channels only.
     */
    std::vector<float> getDelaysApertureOnly() const {
        std::vector<float> txDelays;
        for(size_t i = 0; i < getAperture().size(); ++i) {
            if(getAperture()[i]) {
                txDelays.push_back(getDelays()[i]);
            }
        }
        return txDelays;
    }

    /**
     * Returns true if this operator does not perform TX at all (i.e. aperture is set to false).
     */
    bool isNOP() const {
        bool atLeastOneActive = false;
        for(auto bit: aperture) {
            atLeastOneActive = atLeastOneActive | bit;
        }
        return !atLeastOneActive;
    }


private:
    std::vector<bool> aperture;
    std::vector<float> delays;
    Waveform excitation;
    devices::DeviceId placement;
};

}// namespace arrus::ops::us4r

#endif//ARRUS_CORE_API_OPS_US4R_TX_H
