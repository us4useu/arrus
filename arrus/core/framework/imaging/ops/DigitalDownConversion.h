#ifndef CPP_EXAMPLE_IMAGING_OPS_DIGITALDOWNCONVERSION_H
#define CPP_EXAMPLE_IMAGING_OPS_DIGITALDOWNCONVERSION_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Digital down conversion: demodulate, decimate by a given factor.
 */
class DigitalDownConversion {
public:
    DigitalDownConversion(const NdArray &coeffs, const NdArray &decimationFactor) {
        op = OperationBuilder{}
                 .setClassId(OPERATION_CLASS_ID(DigitalDownConversion))
                 .addParam("coefficients", coeffs)
                 .addParam("decimationFactor", decimationFactor)
                 .build();
    }

    operator Operation() { return op; }

private:
    Operation op;
};

}

#endif//CPP_EXAMPLE_IMAGING_OPS_DIGITALDOWNCONVERSION_H
