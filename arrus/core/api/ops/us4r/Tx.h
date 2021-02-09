#ifndef ARRUS_CORE_API_OPS_US4R_TX_H
#define ARRUS_CORE_API_OPS_US4R_TX_H

#include <utility>

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
	 */
    Tx(std::vector<bool> aperture, std::vector<float> delays, const Pulse &excitation)
        : aperture(std::move(aperture)),
          delays(std::move(delays)),
          excitation(excitation) {}

    const std::vector<bool> &getAperture() const {
        return aperture;
    }

    const std::vector<float> &getDelays() const {
        return delays;
    }

    const Pulse &getExcitation() const {
        return excitation;
    }

private:
    std::vector<bool> aperture;
    std::vector<float> delays;
    Pulse excitation;
};


}

#endif //ARRUS_CORE_API_OPS_US4R_TX_H
