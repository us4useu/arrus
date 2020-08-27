#ifndef ARRUS_API_MATLAB_WRAPPERS_ASSERTS_H
#define ARRUS_API_MATLAB_WRAPPERS_ASSERTS_H

#include <limits>

#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/api/matlab/wrappers/common.h"

#define ARRUS_MATLAB_REQUIRES_N_PARAMETERS(inputs, n, methodName) \
    ARRUS_REQUIRES_EQUAL((inputs).size(), (n),     \
        arrus::IllegalArgumentException(arrus::format( \
            "Function '{}' requires exactly {} parameters (got {})", \
                (methodName), (n), (inputs).size())))


#define ARRUS_MATLAB_REQUIRES_SCALAR(array, msg)          \
do {                                                      \
    if (!::arrus::matlab::isArrayScalar(array)) {         \
        throw arrus::IllegalArgumentException(msg);       \
    }                                                     \
} while(0)

#define ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE(value, dataType)          \
do {                                                                    \
    dataType min = std::numeric_limits<dataType>::min();                \
    dataType max = std::numeric_limits<dataType>::max();                \
    if (value < min || value > max) {                                   \
        throw arrus::IllegalArgumentException(arrus::format(            \
            "Value {} should be in range [{}, {}]", value, min, max     \
        ));                                                             \
    }                                                                   \
} while(0)
#endif //ARRUS_API_MATLAB_WRAPPERS_ASSERTS_H
