#ifndef ARRUS_API_MATLAB_WRAPPERS_CONVERT_H
#define ARRUS_API_MATLAB_WRAPPERS_CONVERT_H

#include <algorithm>
#include <vector>

#include "MexContext.h"
#include "api/matlab/wrappers/asserts.h"
#include "mex.hpp"
#include "mexAdapter.hpp"
// A header with converting routines mex - cpp for basic types: string, matlab,
// etc.

namespace arrus::matlab::converters {

// Utility functions.

::matlab::data::Array getMatlabProperty(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object,
                                        const std::string &propertyName) {
    return ctx->getMatlabEngine()->getProperty(object, propertyName);
}

template<typename Out, typename In> Out safeCast(In value) {
    return Out(value);
}
template<typename Out, typename In> Out safeCastInt(In value) {
    ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE(value, Out);
    ARRUS_MATLAB_REQUIRES_INTEGER(value);
    return Out(value);
}

// Integer values: make sure the values are in proper range.
template<> bool safeCast<bool, double>(double value) {
    if(value != 1.0 && value != 0.0) {
        throw IllegalArgumentException("Value " + std::to_string(value) + " should be 0 or 1 (boolean).");
    }
    return bool(value);
}

template<> uint8_t safeCast<uint8_t, double>(double value) {
    return safeCastInt<uint8_t>(value);
}
template<> int8_t safeCast<int8_t, double>(double value) {
    return safeCastInt<int8_t>(value);
}
template<> int16_t safeCast<int16_t, double>(double value) {
    return safeCastInt<int16_t>(value);
}
template<> uint16_t safeCast<uint16_t, double>(double value) {
    return safeCastInt<uint16_t>(value);
}
template<> int32_t safeCast<int32_t, double>(double value) {
    return safeCastInt<int32_t>(value);
}
template<> uint32_t safeCast<uint32_t, double>(double value) {
    return safeCastInt<uint32_t>(value);
}
template<> int64_t safeCast<int64_t, double>(double value) {
    return safeCastInt<int64_t>(value);
}
template<> uint64_t safeCast<uint64_t>(double value) {
    return safeCastInt<uint64_t>(value);
}

// MATLAB ARRAY -> SCALAR C++ VALUE
std::string convertToString(const ::matlab::data::CharArray &charArray) { return charArray.toAscii(); }

/**
 * Converts a given numeric MATLAB scalar to C++ value. Throws IllegalArgumentException if the given data type
 * is not consistent with the property value.
 *
 * @tparam T type of the output value
 * @param array input array
 * @param arrayName name of the array (will be used in error msg to user if necessary)
 * @return
 */
template<typename T> T convertToCppScalar(const ::matlab::data::Array &array, const std::string &arrayName) {
    ARRUS_MATLAB_REQUIRES_SCALAR(array, arrayName);
    double value = array[0];
    try {
        return safeCast<T>(value);
    } catch(const IllegalArgumentException &e) {
        throw IllegalArgumentException("Error while converting " + arrayName + " to C++ scalar: " + e.what());
    }
}

// convertToCppScalar specializations
template<> std::string convertToCppScalar<std::string>(const ::matlab::data::Array &array, const std::string &arrayName) {
    return convertToString(array);
}


/**
 * Reads property with a given name and converts it to a given numeric MATLAB scalar to C++ value.
 * Throws IllegalArgumentException if the given data type is not consistent with the property value.
 *
 * @tparam T type of the output value
 * @param array input array
 * @param arrayName name of the array (will be used in error msg to user if necessary)
 * @return C++ value
 */
template<typename T>
T getCppScalar(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object, const std::string &propertyName) {
    ::matlab::data::Array arr = getMatlabProperty(ctx, object, propertyName);
    return convertToCppScalar<T>(arr, propertyName);
}

#define ARRUS_MATLAB_GET_CPP_SCALAR(ctx, type, field, arrayScalar) getCppScalar<type>(ctx, arrayScalar, #field)

/**
 * Reads property with a given name and converts it to a given numeric MATLAB scalar to C++ value.
 * Throws IllegalArgumentException if the given data type is not consistent with the property value.
 * Returns std::nullopt if the given property was an empty array.
 *
 * @tparam T type of the output value
 * @param array input array
 * @param arrayName name of the array (will be used in error msg to user if necessary)
 * @return C++ value or std::nullopt if the given property was an empty array
 */
template<typename T>
std::optional<T> getCppOptionalScalar(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object,
                                   const std::string &propertyName) {
    ::matlab::data::Array arr = getMatlabProperty(ctx, object, propertyName);
    if (arr.isEmpty()) {
        return {};
    }
    return convertToCppScalar<T>(arr, propertyName);
}

#define ARRUS_MATLAB_GET_CPP_OPTIONAL_SCALAR(ctx, type, field, arrayScalar) getCppOptionalScalar<type>(ctx, arrayScalar, #field)

/**
 * Reads property with a given name and converts it to a given numeric MATLAB scalar to C++ value.
 * Throws IllegalArgumentException if the given data type is not consistent with the property value, or
 * is an empty array.
 *
 * @tparam T type of the output value
 * @param array input array
 * @param arrayName name of the array (will be used in error msg to user if necessary)
 * @return C++ value
 */
template<typename T>
T getCppRequiredScalar(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object,
                                        const std::string &propertyName) {
    ::matlab::data::Array arr = getMatlabProperty(ctx, object, propertyName);
    if (arr.isEmpty()) {
        throw IllegalArgumentException(arrus::format("Field '{}' is required", propertyName));
    }
    return convertToCppScalar<T>(arr, propertyName);
}

#define ARRUS_MATLAB_GET_CPP_REQUIRED_SCALAR(ctx, type, field, arrayScalar) getCppRequiredScalar<type>(ctx, arrayScalar, #field)

// MATLAB ARRAY -> FLAT std::vector
/**
 * Converts a given numeric MATLAB array to flat std::vector.
 *
 * @tparam T type of the output value
 * @param array input array
 * @param arrayName name of the array (will be used in error msg to user if necessary)
 * @return
 */
template<typename Out> std::vector<Out> convertToCppVector(const ::matlab::data::TypedArray<double> &t,
                                    const std::string &arrayName) {
    std::vector<Out> result(t.getNumberOfElements());
    try {
        for(auto it = std::begin(t); it != std::end(t); ++it) {
            result.push_back(safeCast<Out, double>(*it));
        }
    } catch(const IllegalArgumentException &e) {
        throw IllegalArgumentException("Error while converting " + arrayName + " to C++ vector: " + e.what());
    }
    return result;
}

template<typename T>
std::vector<T> getCppVector(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object,
                         const std::string &propertyName) {
    ::matlab::data::TypedArray<double> arr = getMatlabProperty(ctx, object, propertyName);
    return convertToCppVector<T>(arr);
}

#define ARRUS_MATLAB_GET_CPP_VECTOR(ctx, type, field, arrayScalar) getCppVector<type>(ctx, array, #field)

// C++ scalar -> MATLAB ARRAY
template<typename T>
::matlab::data::TypedArray<T> getMatlabScalar(const MexContext::SharedHandle &ctx, T value) {
    return ctx->createScalar<T>(value);
}

#define ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, type, value) getMatlabScalar<type>(ctx, value)


}// namespace arrus::matlab::converters

#endif//ARRUS_API_MATLAB_WRAPPERS_CONVERT_H
