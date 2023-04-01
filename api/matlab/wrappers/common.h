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
    using MatlabObjectHandle = size_t;
    using MatlabClassId = std::string;

    using MatlabMethodId = std::string;
    using MatlabInputArgs = ::matlab::mex::ArgumentList;
    using MatlabOutputArgs = ::matlab::mex::ArgumentList;


    bool inline isArrayScalar(const ::matlab::data::Array &array) {
        return array.getNumberOfElements() == 1;
    }

    bool inline isArrayOfType(const ::matlab::data::Array &array, ::matlab::data::ArrayType type) {
        return array.getType() == type;
    }

    std::string toString(::matlab::data::ArrayType type) {
        switch(type) {
        case ::matlab::data::ArrayType::LOGICAL:
            return "Logical";
        case ::matlab::data::ArrayType::MATLAB_STRING:
            return "MATLAB string";
        case ::matlab::data::ArrayType::DOUBLE:
            return "double";
        case ::matlab::data::ArrayType::SINGLE:
            return "single";
        case ::matlab::data::ArrayType::UINT8:
            return "uint8";
        case ::matlab::data::ArrayType::INT8:
            return "int8";
        case ::matlab::data::ArrayType::UINT16:
            return "uint16";
        case ::matlab::data::ArrayType::INT16:
            return "int16";
        case ::matlab::data::ArrayType::UINT32:
            return "uint32";
        case ::matlab::data::ArrayType::INT32:
            return "int32";
        case ::matlab::data::ArrayType::UINT64:
            return "uint64";
        case ::matlab::data::ArrayType::INT64:
            return "int64";
        case ::matlab::data::ArrayType::CHAR:
            return "char";
        case ::matlab::data::ArrayType::COMPLEX_DOUBLE:
            return "complex double";
        case ::matlab::data::ArrayType::COMPLEX_SINGLE:
            return "complex single";
        case ::matlab::data::ArrayType::COMPLEX_UINT8:
            return "complex uint8";
        case ::matlab::data::ArrayType::COMPLEX_INT8:
            return "complex int8";
        case ::matlab::data::ArrayType::COMPLEX_UINT16:
            return "complex uint16";
        case ::matlab::data::ArrayType::COMPLEX_INT16:
            return "complex int16";
        case ::matlab::data::ArrayType::COMPLEX_UINT32:
            return "complex uint32";
        case ::matlab::data::ArrayType::COMPLEX_INT32:
            return "complex int32";
        case ::matlab::data::ArrayType::COMPLEX_UINT64:
            return "complex uint64";
        case ::matlab::data::ArrayType::COMPLEX_INT64:
            return "complex int64";
        case ::matlab::data::ArrayType::SPARSE_COMPLEX_DOUBLE:
            return "sparse complex double";
        case ::matlab::data::ArrayType::SPARSE_DOUBLE:
            return "sparse double";
        case ::matlab::data::ArrayType::SPARSE_LOGICAL:
            return "sparse logical";
        case ::matlab::data::ArrayType::CELL:
            return "cell";
        case ::matlab::data::ArrayType::OBJECT:
            return "object";
        case ::matlab::data::ArrayType::ENUM:
            return "enum";
        case ::matlab::data::ArrayType::STRUCT:
            return "struct";
        case ::matlab::data::ArrayType::VALUE_OBJECT:
            return "value object";
        case ::matlab::data::ArrayType::HANDLE_OBJECT_REF:
            return "handle object ref";
        case ::matlab::data::ArrayType::UNKNOWN:
            return "unknown";
        default:
            throw std::runtime_error("Unknown MATLAB array type: " + std::to_string((size_t)type));
        }
    }

    bool inline isInteger(const double value) {
        double ignore;
        return std::modf(value, &ignore) == 0.0;
    }
}

#endif //ARRUS_API_MATLAB_WRAPPERS_COMMON_H
