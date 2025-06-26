#ifndef ARRUS_API_MATLAB_WRAPPERS_CONVERT_H
#define ARRUS_API_MATLAB_WRAPPERS_CONVERT_H

#include <algorithm>
#include <vector>

#include "MexContext.h"
#include "api/matlab/wrappers/asserts.h"
#include "arrus/core/api/common.h"
#include "mex.hpp"
#include "mexAdapter.hpp"
// A header with converting routines mex - cpp for basic types: string, matlab,
// etc.

namespace arrus::matlab::converters {

// Utility functions.

/**
 * A pair: MATLAB array, index.
 * This pair provides the possibility to read property of a particular object array element.
 * This is because the only way to get i-th object's property from an array of objects is to use:
 * matlab::data::Array getProperty(const matlab::data::Array &objectArray,
 * size_t index,
 * const matlab::engine::String &propertyName)
 */
struct MatlabElementRef {
    MatlabElementRef(const ::matlab::data::Array &array, size_t i) : array(array), i(i) {}
    explicit MatlabElementRef(const ::matlab::data::Array &array) : array(array), i(0) {}

    ::matlab::data::Array array;
    size_t i;
};

using ::matlab::data::ArrayType;

bool isMatlabLogical(ArrayType type) { return type == ArrayType::LOGICAL; }

bool isMatlabInteger(ArrayType type) {
    return type == ArrayType::UINT8 || type == ArrayType::INT8 || type == ArrayType::UINT16 || type == ArrayType::INT16
        || type == ArrayType::UINT32 || type == ArrayType::INT32 || type == ArrayType::UINT64
        || type == ArrayType::INT64;
}

bool isMatlabFloatingPoint(ArrayType type) { return type == ArrayType::DOUBLE || type == ArrayType::SINGLE; }

bool isMatlabRealNumeric(ArrayType type) {
    return isMatlabLogical(type) || isMatlabInteger(type) || isMatlabFloatingPoint(type);
}

bool isMatlabString(ArrayType type) { return type == ArrayType::MATLAB_STRING; }

::matlab::data::Array getMatlabProperty(const MexContext::SharedHandle &ctx, const MatlabElementRef &object,
                                        const std::string &propertyName) {
    return ctx->getMatlabEngine()->getProperty(object.array, object.i, propertyName);
}

template<typename T> T safeCast(const ::matlab::data::Array &arr, const size_t i) { return T(arr[i]); }

template<typename T> T safeCastInt(const ::matlab::data::Array &arr, const size_t i) {
    auto type = arr.getType();
    if (isMatlabLogical(type) || isMatlabInteger(type)) {
        T v = arr[i];
        return v;
    } else if (isMatlabFloatingPoint(type)) {
        double v = arr[i];
        auto min = (double)std::numeric_limits<T>::min();
        auto max = (double)std::numeric_limits<T>::max();
        if (v < min || v > max) {
            arrus::IllegalArgumentException(arrus::format("Value {} should be in range [{}, {}]", v, min, max));
        }
        return v;
    } else {
        throw ::arrus::IllegalArgumentException("Unsupported data type for the integer output.");
    }
}

// floating-point -> integer safe cast
template<> bool safeCast<bool>(const ::matlab::data::Array &arr, const size_t i) { return safeCastInt<bool>(arr, i); }
template<> uint8_t safeCast<uint8_t>(const ::matlab::data::Array &arr, const size_t i) {
    return safeCastInt<uint8_t>(arr, i);
}
template<> int8_t safeCast<int8_t>(const ::matlab::data::Array &arr, const size_t i) {
    return safeCastInt<int8_t>(arr, i);
}
template<> uint16_t safeCast<uint16_t>(const ::matlab::data::Array &arr, const size_t i) {
    return safeCastInt<uint16_t>(arr, i);
}
template<> int16_t safeCast<int16_t>(const ::matlab::data::Array &arr, const size_t i) {
    return safeCastInt<int16_t>(arr, i);
}
template<> uint32_t safeCast<uint32_t>(const ::matlab::data::Array &arr, const size_t i) {
    return safeCastInt<uint32_t>(arr, i);
}
template<> int32_t safeCast<int32_t>(const ::matlab::data::Array &arr, const size_t i) {
    return safeCastInt<int32_t>(arr, i);
}
template<> uint64_t safeCast<uint64_t>(const ::matlab::data::Array &arr, const size_t i) {
    return safeCastInt<uint64_t>(arr, i);
}
template<> int64_t safeCast<int64_t>(const ::matlab::data::Array &arr, const size_t i) {
    return safeCastInt<int64_t>(arr, i);
}

// MATLAB ARRAY -> SCALAR C++ VALUE
std::string convertToString(const ::matlab::data::StringArray &stringArray) { return stringArray[0]; }

// Functions that allow to verify, if the array data type is compatible with the expected (compile-time) type.

template<typename T> bool isArrayTypeOkFor(ArrayType arrayType) { return false; }
template<> bool isArrayTypeOkFor<::arrus::float64>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::float32>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::int64>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::uint64>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::int32>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::uint32>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::int16>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::uint16>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::int8>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::uint8>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<::arrus::boolean>(ArrayType arrayType) { return isMatlabRealNumeric(arrayType); }
template<> bool isArrayTypeOkFor<std::string>(ArrayType arrayType) { return isMatlabString(arrayType); }

template<typename T> bool isMatlabArrayCompatibleWithType(const ::matlab::data::Array &array) {
    try {
        return isArrayTypeOkFor<T>(array.getType());
    } catch (const std::exception &e) {
        throw ::arrus::IllegalArgumentException(
            ::arrus::format("Exception while resolving array's data type: {}", e.what()));
    }
}
#define ARRUS_MATLAB_REQUIRES_COMPATIBLE_TYPE_FOR_PROPERTY(array, expectedType, propertyName)                          \
    do {                                                                                                               \
        if (!isMatlabArrayCompatibleWithType<expectedType>(array)) {                                                   \
            throw arrus::IllegalArgumentException("Incompatible expected and given type for: " + (propertyName));      \
        }                                                                                                              \
    } while (0)

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
    try {
        return safeCast<T>(array, 0);
    } catch (const IllegalArgumentException &e) {
        throw IllegalArgumentException("Error while converting " + arrayName + " to C++ scalar: " + e.what());
    }
}

// convertToCppScalar specializations
template<>
std::string convertToCppScalar<std::string>(const ::matlab::data::Array &array, const std::string &arrayName) {
    std::string result = convertToString(array);
    return result;
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
T getCppScalar(const MexContext::SharedHandle &ctx, const MatlabElementRef &object, const std::string &propertyName) {
    try {
        ::matlab::data::Array arr = getMatlabProperty(ctx, object, propertyName);
        ARRUS_MATLAB_REQUIRES_SCALAR_NAMED(arr, propertyName);
        ARRUS_MATLAB_REQUIRES_COMPATIBLE_TYPE_FOR_PROPERTY(arr, T, propertyName);
        return convertToCppScalar<T>(arr, propertyName);
    } catch (const std::exception &e) {
        throw ::arrus::IllegalArgumentException(
            ::arrus::format("Exception while getting property '{}' to C++: {}", propertyName, e.what()));
    }
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
std::optional<T> getCppOptionalScalar(const MexContext::SharedHandle &ctx, const MatlabElementRef &object,
                                      const std::string &propertyName) {
    try {
        ::matlab::data::Array arr = getMatlabProperty(ctx, object, propertyName);
        if (arr.isEmpty()) {
            return {};
        }
        ARRUS_MATLAB_REQUIRES_SCALAR_NAMED(arr, propertyName);
        ARRUS_MATLAB_REQUIRES_COMPATIBLE_TYPE_FOR_PROPERTY(arr, T, propertyName);
        return convertToCppScalar<T>(arr, propertyName);
    } catch (const std::exception &e) {
        throw ::arrus::IllegalArgumentException(
            ::arrus::format("Exception while reading property '{}' to C++: {}", propertyName, e.what()));
    }
}

#define ARRUS_MATLAB_GET_CPP_OPTIONAL_SCALAR(ctx, type, field, arrayScalar)                                            \
    getCppOptionalScalar<type>(ctx, arrayScalar, #field)

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
T getCppRequiredScalar(const MexContext::SharedHandle &ctx, const MatlabElementRef &object,
                       const std::string &propertyName) {
    try {
        ::matlab::data::Array arr = getMatlabProperty(ctx, object, propertyName);
        if (arr.isEmpty()) {
            throw IllegalArgumentException(arrus::format("Field '{}' is required", propertyName));
        }
        ARRUS_MATLAB_REQUIRES_SCALAR_NAMED(arr, propertyName);
        ARRUS_MATLAB_REQUIRES_COMPATIBLE_TYPE_FOR_PROPERTY(arr, T, propertyName);
        return convertToCppScalar<T>(arr, propertyName);
    } catch (const std::exception &e) {
        throw ::arrus::IllegalArgumentException(
            ::arrus::format("Exception while reading property '{}' to C++: {}", propertyName, e.what()));
    }
}

#define ARRUS_MATLAB_GET_CPP_REQUIRED_SCALAR(ctx, type, field, arrayScalar)                                            \
    getCppRequiredScalar<type>(ctx, arrayScalar, #field)

// MATLAB ARRAY -> FLAT std::vector
/**
 * Converts a given numeric MATLAB array to flat std::vector.
 *
 * @tparam T type of the output value
 * @param array input array
 * @param arrayName name of the array (will be used in error msg to user if necessary)
 * @return
 */
template<typename T> std::vector<T> convertToCppVector(const ::matlab::data::Array &t, const std::string &arrayName) {
    std::vector<T> result(t.getNumberOfElements());
    try {
        for (int i = 0; i < t.getNumberOfElements(); ++i) {
            result[i] = safeCast<T>(t, i);
        }
    } catch (const IllegalArgumentException &e) {
        throw IllegalArgumentException("Error while converting " + arrayName + " to C++ vector: " + e.what());
    }
    return result;
}

template<typename T>
std::vector<T> getCppVector(const MexContext::SharedHandle &ctx, const MatlabElementRef &object,
                            const std::string &propertyName) {
    try {
        ::matlab::data::Array arr = getMatlabProperty(ctx, object, propertyName);
        ARRUS_MATLAB_REQUIRES_COMPATIBLE_TYPE_FOR_PROPERTY(arr, T, propertyName);
        return convertToCppVector<T>(arr, propertyName);
    } catch (const std::exception &e) {
        throw ::arrus::IllegalArgumentException(
            ::arrus::format("Exception while reading property '{}' to C++: {}", propertyName, e.what()));
    }
}

#define ARRUS_MATLAB_GET_CPP_VECTOR(ctx, type, field, array) getCppVector<type>(ctx, array, #field)

template<typename T>
std::pair<T, T> getCppPair(const MexContext::SharedHandle &ctx, const MatlabElementRef &object,
                           const std::string &propertyName) {
    std::vector<T> vec = getCppVector<T>(ctx, object, propertyName);
    if (vec.size() != 2) {
        throw ::arrus::IllegalArgumentException(::arrus::format("Exception while reading property '{} to C++: "
                                                                "the array should contains exactly two values.",
                                                                propertyName));
    }
    return std::make_pair(vec[0], vec[1]);
}

#define ARRUS_MATLAB_GET_CPP_PAIR(ctx, type, field, array) getCppPair<type>(ctx, array, #field)

// MATLAB OBJECT -> C++ object
template<typename T, typename Converter>
std::vector<T> getCppObjectVector(const MexContext::SharedHandle &ctx, const MatlabElementRef &object,
                                  const std::string &property) {
    try {
        std::vector<T> result;
        // e.g. seq -> ops: an array of ops
        ::matlab::data::Array array = getMatlabProperty(ctx, object, property);
        if(array.isEmpty()) {
            return result;
        }
        ::matlab::data::ObjectArray objArray = array;
        for (size_t i = 0; i < objArray.getNumberOfElements(); ++i) {
            MatlabElementRef ref{objArray, i};
            result.emplace_back(Converter::from(ctx, ref).toCore());
        }
        return result;
    } catch (const std::exception &e) {
        throw ::arrus::IllegalArgumentException(
            ::arrus::format("Exception while reading property '{}' to C++: {}", property, e.what()));
    }
}

template<typename T, typename Converter>
std::optional<T> getCppOptionalObject(const MexContext::SharedHandle &ctx, const MatlabElementRef &object,
                                      const std::string &property) {
    std::vector<T> values = getCppObjectVector<T, Converter>(ctx, object, property);
    if (values.empty()) {
        return {};
    } else
        return values[0];
}

template<typename T, typename Converter>
T getCppObject(const MexContext::SharedHandle &ctx, const MatlabElementRef &object, const std::string &property) {
    return getCppObjectVector<T, Converter>(ctx, object, property)[0];
}

#define ARRUS_MATLAB_GET_CPP_OBJECT(ctx, Type, Converter, field, array)                                                \
    getCppObject<Type, Converter>(ctx, array, #field)

#define ARRUS_MATLAB_GET_CPP_OPTIONAL_OBJECT(ctx, Type, Converter, field, array)                                       \
    getCppOptionalObject<Type, Converter>(ctx, array, #field)

#define ARRUS_MATLAB_GET_CPP_OBJECT_VECTOR(ctx, Type, Converter, field, array)                                         \
    getCppObjectVector<Type, Converter>(ctx, array, #field)

// C++ scalar -> MATLAB ARRAY
template<typename T>
::matlab::data::Array getMatlabScalar(const MexContext::SharedHandle &ctx, T value) {
    return ctx->createScalar<T>(value);
}

template<typename T>
::matlab::data::Array getMatlabScalar(const MexContext::SharedHandle &ctx, const std::optional<T> &t) {
    if(!t.has_value()) {
        return ctx->getArrayFactory().createEmptyArray();
    }
    else {
        return getMatlabScalar<T>(ctx, t.value());
    }
}

::matlab::data::TypedArray<::matlab::data::MATLABString> getMatlabString(const MexContext::SharedHandle &ctx,
                                                                         const char16_t *value) {
    return ctx->createScalarString(::matlab::data::MATLABString{value});
}

::matlab::data::TypedArray<::matlab::data::MATLABString> getMatlabString(const MexContext::SharedHandle &ctx,
                                                                         const std::string &str) {
    return ctx->createScalarString(str);
}

#define ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, type, value) getMatlabScalar<type>(ctx, value)
// Creates pair: key, value, key will be determined by value keyword.
#define ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, type, value)                                                            \
    getMatlabString(ctx, u## #value), ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, type, value)

#define ARRUS_MATLAB_GET_MATLAB_STRING(ctx, value) getMatlabString(ctx, value)
#define ARRUS_MATLAB_GET_MATLAB_STRING_KV_EXPLICIT(ctx, key, value)                                                    \
    getMatlabString(ctx, key), ARRUS_MATLAB_GET_MATLAB_STRING(ctx, value)
#define ARRUS_MATLAB_GET_MATLAB_STRING_KV(ctx, value)                                                                  \
    getMatlabString(ctx, u## #value), ARRUS_MATLAB_GET_MATLAB_STRING(ctx, value)

// C++ std::vector/pair -> MATLAB ARRAY
template<typename T>
::matlab::data::TypedArray<T> getMatlabVector(const MexContext::SharedHandle &ctx, std::vector<T> values) {
    return ctx->createVector<T>(values);
}

template<typename T>
::matlab::data::TypedArray<T> getMatlabVector(const MexContext::SharedHandle &ctx, std::pair<T, T> values) {
    std::vector<T> vec = {values.first, values.second};
    return ctx->createVector<T>(vec);
}

#define ARRUS_MATLAB_GET_MATLAB_VECTOR(ctx, type, value) getMatlabVector<type>(ctx, value)
// Produces pair: key, value, key will be determined by value keyword.
#define ARRUS_MATLAB_GET_MATLAB_VECTOR_KV_EXPLICIT(ctx, type, key, value)                                                            \
    getMatlabString(ctx, u## #key), ARRUS_MATLAB_GET_MATLAB_VECTOR(ctx, type, value)
#define ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, type, value)                                                            \
    getMatlabString(ctx, u## #value), ARRUS_MATLAB_GET_MATLAB_VECTOR(ctx, type, value)


// C++ Object -> MATLAB ARRAY
template<typename T, typename Converter>
::matlab::data::Array getMatlabObject(const MexContext::SharedHandle &ctx, const T &t) {
    return Converter::from(ctx, t).toMatlab();
}


template<typename T, typename Converter>
::matlab::data::Array getMatlabObject(const MexContext::SharedHandle &ctx, const std::optional<T> &t) {
    if(!t.has_value()) {
        return ctx->getArrayFactory().createEmptyArray();
    }
    else {
        return getMatlabObject<T, Converter>(ctx, t.value());
    }
}

#define ARRUS_MATLAB_GET_MATLAB_OBJECT(ctx, Type, Converter, value) getMatlabObject<Type, Converter>(ctx, value)
// Produces pair: key, value, key will be determined by value keyword.
#define ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, Type, Converter, value)                                                 \
    getMatlabString(ctx, u## #value), ARRUS_MATLAB_GET_MATLAB_OBJECT(ctx, Type, Converter, value)

template<typename T, typename Converter>
::matlab::data::Array getMatlabObjectVector(const MexContext::SharedHandle &ctx, const std::vector<T> &t) {
    ::matlab::data::ArrayDimensions dims{1, t.size()};
    std::vector<::matlab::data::Object> objects;
    for (int i = 0; i < t.size(); ++i) {
        const auto &value = t[i];
        ::matlab::data::ObjectArray arr = Converter::from(ctx, value).toMatlab();
        ::matlab::data::Object object = arr[0];
        objects.emplace_back(object);
    }
    return ctx->getArrayFactory().createArray(dims, std::begin(objects), std::end(objects));
}

#define ARRUS_MATLAB_GET_MATLAB_OBJECT_VECTOR(ctx, Type, Converter, value)                                             \
    getMatlabObjectVector<Type, Converter>(ctx, value)
// Produces pair: key, value, key will be determined by value keyword.
#define ARRUS_MATLAB_GET_MATLAB_OBJECT_VECTOR_KV(ctx, Type, Converter, value)                                          \
    getMatlabString(ctx, u## #value), ARRUS_MATLAB_GET_MATLAB_OBJECT_VECTOR(ctx, Type, Converter, value)

}// namespace arrus::matlab::converters

#endif//ARRUS_API_MATLAB_WRAPPERS_CONVERT_H
