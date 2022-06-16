#ifndef CPP_EXAMPLE_IMAGING_OPS_TOBMODE_H
#define CPP_EXAMPLE_IMAGING_OPS_TOBMODE_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Beamforms input RF data to a sequence of high-resolution images
 * (a sum of low-resolution images).
 * The output images will reconstructed on a given grid of pixels.
 *
 * Currently works only with plane wave data.
 *
 */
class ToBMode {
public:
    ToBMode(const NdArray &minDbLimit, const NdArray &maxDbLimit) {
        op = OperationBuilder{}
                 .setClassId(OPERATION_CLASS_ID(ToBMode))
                 .addParam("minDbLimit", minDbLimit)
                 .addParam("maxDbLimit", maxDbLimit)
                 .build();
    }

    operator Operation() { return op; }

private:
    Operation op;
};

}


#endif//CPP_EXAMPLE_IMAGING_OPS_TOBMODE_H
