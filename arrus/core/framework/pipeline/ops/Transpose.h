#ifndef CPP_EXAMPLE_IMAGING_OPS_TRANSPOSE_H
#define CPP_EXAMPLE_IMAGING_OPS_TRANSPOSE_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Transposes by the last two axes of array.
 */
class Transpose {
public:
    Transpose() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(Transpose)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}
#endif//CPP_EXAMPLE_IMAGING_OPS_TRANSPOSE_H
