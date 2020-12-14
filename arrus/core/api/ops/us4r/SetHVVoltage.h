#ifndef ARRUS_ARRUS_CORE_API_OPS_US4R_SETHVVOLTAGE_H
#define ARRUS_ARRUS_CORE_API_OPS_US4R_SETHVVOLTAGE_H

#include "arrus/core/api/ops/Op.h"

namespace arrus::ops::us4r {

/**
 * An operation to set HV voltage on the device asynchronously.
 */
class SetHVVoltage : Op {
public:
    explicit SetHVVoltage(unsigned char voltage) : voltage(voltage) {}

    [[nodiscard]] unsigned char getVoltage() const {
        return voltage;
    }

    unsigned getTypeId() override {
        return opTypeId;
    }

private:
    static unsigned int opTypeId;
    unsigned char voltage;
};

// TODO type id counter or op name + hash
unsigned SetHVVoltage::opTypeId = 0;

}


#endif //ARRUS_ARRUS_CORE_API_OPS_US4R_SETHVVOLTAGE_H
