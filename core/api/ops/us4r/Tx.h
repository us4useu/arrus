#ifndef ARRUS_CORE_API_OPS_US4R_TX_H
#define ARRUS_CORE_API_OPS_US4R_TX_H

#include <utility>

#include "Pulse.h"
#include "arrus/core/api/common/types.h"

namespace arrus::ops::us4r {

class Tx {
public:
    Tx(BitMask aperture, std::vector<float> delays,
       const Pulse &excitation) : aperture(std::move(aperture)),
                                  delays(std::move(delays)),
                                  excitation(excitation) {}

    [[nodiscard]] const BitMask &getAperture() const {
        return aperture;
    }

    [[nodiscard]] const std::vector<float> &getDelays() const {
        return delays;
    }

    [[nodiscard]] const Pulse &getExcitation() const {
        return excitation;
    }

private:
    BitMask aperture;
    std::vector<float> delays;
    Pulse excitation;
};


}

#endif //ARRUS_CORE_API_OPS_US4R_TX_H