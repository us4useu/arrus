#ifndef ARRUS_CORE_API_OPS_US4R_TX_H
#define ARRUS_CORE_API_OPS_US4R_TX_H

#include <utility>

#include "arrus/core/api/devices/DeviceId.h"
#include "Pulse.h"
#include "arrus/core/api/common/types.h"

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
	 * @param excitation pulse to transmit
	 * @parma placement probe on which the Tx should be performed
	 */
    Tx(std::vector<bool> aperture, std::vector<float> delays, const Pulse &excitation, devices::DeviceId placement)
        : aperture(std::move(aperture)),
          delays(std::move(delays)),
          excitation(excitation),
          placement(placement){}

    Tx(std::vector<bool> aperture, std::vector<float> delays, const Pulse &excitation)
        : aperture(std::move(aperture)),
          delays(std::move(delays)),
          excitation(excitation),
          placement(devices::DeviceId(devices::DeviceType::Probe, 0)){}

    const std::vector<bool> &getAperture() const { return aperture; }

    const std::vector<float> &getDelays() const { return delays; }

    const Pulse &getExcitation() const { return excitation; }

    const devices::DeviceId &getPlacement() const { return placement; }

private:
    std::vector<bool> aperture;
    std::vector<float> delays;
    Pulse excitation;
    devices::DeviceId placement;
};


}

#endif //ARRUS_CORE_API_OPS_US4R_TX_H
