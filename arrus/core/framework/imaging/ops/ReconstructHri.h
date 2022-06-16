#ifndef CPP_EXAMPLE_IMAGING_OPS_RECONSTRUCTHRI_H
#define CPP_EXAMPLE_IMAGING_OPS_RECONSTRUCTHRI_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Beamforms input RF data to a sequence of high-resolution images
 * (a sum of low-resolution images).
 * The output images will reconstructed on a given grid of pixels.
 *
 * Currently works only with plane wave data.
 */
class ReconstructHri {
public:
    ReconstructHri(const NdArray &xGrid, const NdArray &zGrid) {
        op = OperationBuilder{}
                 .setClassId(OPERATION_CLASS_ID(ReconstructHri))
                 .addParam("xGrid", xGrid)
                 .addParam("zGrid", zGrid)
                 .build();
    }

    operator Operation() { return op; }

private:
    Operation op;
};

}

#endif//CPP_EXAMPLE_IMAGING_OPS_RECONSTRUCTHRI_H
