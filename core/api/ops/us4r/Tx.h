#ifndef ARRUS_CORE_API_OPS_US4R_TX_H
#define ARRUS_CORE_API_OPS_US4R_TX_H

#include <utility>

#include "Pulse.h"
#include "arrus/core/api/common/types.h"

namespace arrus::ops::us4r {

class Tx {

public:
    Tx(BitMask aperture, std::vector<float> delays, const Pulse &pulse)
        : aperture(std::move(aperture)), delays(std::move(delays)),
          pulse(pulse) {}

private:
    BitMask aperture;
    std::vector<float> delays;
    Pulse pulse;
};


}

#endif //ARRUS_CORE_API_OPS_US4R_TX_H
