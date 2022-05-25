#ifndef ARRUS_API_MATLAB_WRAPPERS_ASSERTS_H
#define ARRUS_API_MATLAB_WRAPPERS_ASSERTS_H

#include <limits>
#include <cmath>

#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "api/matlab/wrappers/common.h"

#define ARRUS_MATLAB_REQUIRES_N_PARAMETERS(inputs, n, methodName) \
    ARRUS_REQUIRES_EQUAL((inputs).size(), (n),     \
        arrus::IllegalArgumentException(arrus::format( \
            "Function '{}' requires exactly {} parameters (got {})", \
                (methodName), (n), (inputs).size())))


#define ARRUS_MATLAB_REQUIRES_SCALAR(array, arrayName)          \
do {                                                      \
    if (!::arrus::matlab::isArrayScalar(array)) {         \
        throw arrus::IllegalArgumentException(arrayName + " should be scalar.");       \
    }                                                     \
} while(0)

#define ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE_EXCEPTION(value, dataType, e)          \
do {                                                                    \
    dataType min = std::numeric_limits<dataType>::min();                \
    dataType max = std::numeric_limits<dataType>::max();                \
    if (value < min || value > max) {                                                \
        throw e; \
    }                                                                   \
} while(0)

#define ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE(value, dataType) \
    ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE_EXCEPTION(value, dataType, \
        arrus::IllegalArgumentException(arrus::format(            \
            "Value {} should be in range [{}, {}]", value, min, max     \
        ));                                                             \
    )

#define ARRUS_MATLAB_REQUIRES_INTEGER_EXCEPTION(value, exception)          \
do {                                                  \
    double ignore;                                    \
    if (std::modf(value, &ignore) != 0.0) {                          \
        throw exception;                                                  \
    }                                                                   \
} while(0)

#define ARRUS_MATLAB_REQUIRES_INTEGER(value) \
    ARRUS_MATLAB_REQUIRES_INTEGER_EXCEPTION( \
        value, arrus::IllegalArgumentException(arrus::format(            \
            "Value {} should be integer", value)))


#define ARRUS_REQUIRES_ALL_DATA_TYPE_VALUE(list, dataType, msg)          \
do {                                                                    \
    dataType min = std::numeric_limits<dataType>::min();                \
    dataType max = std::numeric_limits<dataType>::max();     \
    for(auto value : list) {                                      \
        if (value < min || value > max) {                                   \
            throw arrus::IllegalArgumentException(arrus::format(            \
                "Value {} should be in range [{}, {}], {}",  \
                value, min, max, msg     \
            ));                                                             \
        }                                                    \
    } \
} while(0)

#define ARRUS_MATLAB_REQUIRES_ALL_INTEGER(list)          \
do {                                                  \
    double ignore;                                       \
    for(auto value : list) {                                               \
        if (std::modf(value, &ignore) != 0.0) {   \
            throw arrus::IllegalArgumentException(arrus::format(            \
                "Value {} should be integer", value     \
            ));                                                             \
        }                                                    \
    }\
} while(0)

#define ARRUS_MATLAB_REQUIRES_ALL_BINARY(list)          \
do {                                                    \
    double ignore;\
    for(auto value : list) {                                               \
        if (std::modf(value, &ignore) != 0.0) {   \
            throw arrus::IllegalArgumentException(arrus::format(            \
                "Value {} should be binary", value     \
            ));                                                             \
        }                                               \
        if(value != 1 && value != 0) {                   \
        throw arrus::IllegalArgumentException(arrus::format(            \
                "Value {} should be binary", value     \
            ));                                                         \
        }                                                    \
    }\
} while(0)

#endif //ARRUS_API_MATLAB_WRAPPERS_ASSERTS_H

