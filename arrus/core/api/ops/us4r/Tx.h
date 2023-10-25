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

    Tx(std::vector<bool> aperture, const std::string& delaysProfileName, const Pulse &excitation)
        : aperture(std::move(aperture)), excitation(excitation), delaysProfileName(delaysProfileName) {}

    const std::vector<bool> &getAperture() const {
        return aperture;
    }

    const std::vector<float> &getDelays() const {
        if(!delays.has_value()) {
            ::arrus::IllegalArgumentException("This TX object is using pre-defined TX delays, therefore you are not "
                                              "allowed to use getTxDelays() method.");
        }
        return delays.value();
    }

    const Pulse &getExcitation() const {
        return excitation;
    }

private:
    std::vector<bool> aperture;
    std::optional<std::vector<float>> delays;
    Pulse excitation;
    std::optional<std::string> delaysProfileName;
};


}

#endif //ARRUS_CORE_API_OPS_US4R_TX_H
