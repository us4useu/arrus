#ifndef ARRUS_API_MATLAB_WRAPPERS_CONVERT_H
#define ARRUS_API_MATLAB_WRAPPERS_CONVERT_H

#include <vector>
#include <algorithm>

#include "mex.hpp"
#include "mexAdapter.hpp"
// A header with converting routines mex - cpp for basic types: string, matlab,
// etc.

namespace arrus::matlab {

std::string convertToString(const ::matlab::data::CharArray &charArray) {
    return charArray.toAscii();
}

template<typename Out, typename In>
std::vector<Out> convertToVector(const ::matlab::data::TypedArray<In> &t) {
    std::vector<Out> result(t.getNumberOfElements());
    std::transform(std::begin(t), std::end(t), std::begin(result),
                   [](In value) { return Out(value); });
    return result;
}

template<typename T>
T convertToIntScalar(const ::matlab::data::Array &array,
                     const std::string &arrayName) {
    ARRUS_MATLAB_REQUIRES_SCALAR(array, arrayName);
    double value = array[0];
    ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE(value, T);
    ARRUS_MATLAB_REQUIRES_INTEGER(value);
    return T(value);
}

::matlab::data::Array getProperty(const MexContext::SharedHandle &ctx,
                                  const ::matlab::data::Array &object,
                                  const std::string &propertyName) {
    return ctx->getMatlabEngine()->getProperty(object, propertyName);
}

template<typename T>
T getIntScalar(const MexContext::SharedHandle &ctx,
               const ::matlab::data::Array &object,
               const std::string &propertyName) {
    ::matlab::data::Array arr = getProperty(ctx, object, propertyName);
    return convertToIntScalar<T>(arr, propertyName);
}

template<typename T>
std::optional<T> getOptionalIntScalar(const MexContext::SharedHandle &ctx,
                                      const ::matlab::data::Array &object,
                                      const std::string &propertyName) {
    ::matlab::data::Array arr = getProperty(ctx, object, propertyName);
    if(arr.isEmpty()) {
        return {};
    }
    return convertToIntScalar<T>(arr, propertyName);
}

::matlab::data::Array getRequiredScalar(const MexContext::SharedHandle &ctx,
                                        const ::matlab::data::Array &object,
                                        const std::string &propertyName) {
    ::matlab::data::Array arr = getProperty(ctx, object, propertyName);
    if(arr.isEmpty()) {
        throw IllegalArgumentException(
            arrus::format("Field '{}' is required", propertyName));
    }
    ARRUS_MATLAB_REQUIRES_SCALAR(arr, arrus::format(
        "Field '{}' should be scalar", propertyName));
    return arr;
}

template<typename T>
std::vector<T> getVector(const MexContext::SharedHandle &ctx,
                         const ::matlab::data::Array &object,
                         const std::string &propertyName) {
    ::matlab::data::TypedArray<double> arr =
        getProperty(ctx, object, propertyName);
    ARRUS_REQUIRES_ALL_DATA_TYPE_VALUE(arr, T, propertyName);
    return convertToVector<T>(arr);
}

}


#endif //ARRUS_API_MATLAB_WRAPPERS_CONVERT_H
