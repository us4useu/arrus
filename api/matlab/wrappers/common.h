#ifndef ARRUS_API_MATLAB_WRAPPERS_COMMON_H
#define ARRUS_API_MATLAB_WRAPPERS_COMMON_H

#include <cstdint>
#include <string>
#include <cmath>

#include <arrus/common/compiler.h>

COMPILER_PUSH_DIAGNOSTIC_STATE
#pragma warning(disable: 4100 4189 4458 4702)

#include <mex.hpp>
#include <mexAdapter.hpp>

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::matlab {
    using MexObjectHandle = size_t;
    using MexObjectMethodId = std::string;
    using MexObjectClassId = std::string;

    using MexMethodArgs = ::matlab::mex::ArgumentList;
    using MexMethodReturnType = ::matlab::data::TypedArray<::matlab::data::Array>;


    bool inline isArrayScalar(const ::matlab::data::Array &array) {
        return array.getNumberOfElements() == 1;
    }

    bool inline isInteger(const double value) {
        double ignore;
        return std::modf(value, &ignore) == 0.0;
    }
}

#endif //ARRUS_API_MATLAB_WRAPPERS_COMMON_H
