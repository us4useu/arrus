#ifndef ARRUS_CORE_API_FRAMEWORK_DTYPE_H
#define ARRUS_CORE_API_FRAMEWORK_DTYPE_H

#include <cstddef>
#include <cstdint>

#include "arrus/core/api/common/exceptions.h"

namespace arrus::framework {

/**
 * Numeric element type of an NdArray or NdStorage.
 *
 * Enum values are stable and used as the on-disk serialization codes
 * (see the design doc, Phase 3). Never renumber; new dtypes append at
 * the end.
 */
enum class DType : uint32_t {
    UNKNOWN    = 0,
    BOOL       = 1,
    INT8       = 2,
    UINT8      = 3,
    INT16      = 4,
    UINT16     = 5,
    INT32      = 6,
    UINT32     = 7,
    INT64      = 8,
    UINT64     = 9,
    FLOAT32    = 10,
    FLOAT64    = 11,
    COMPLEX64  = 12,
    COMPLEX128 = 13,
};

/** Size, in bytes, of a single element of the given dtype. */
inline size_t dtypeSize(DType dtype) {
    switch (dtype) {
        case DType::BOOL:       return sizeof(bool);
        case DType::INT8:
        case DType::UINT8:      return 1;
        case DType::INT16:
        case DType::UINT16:     return 2;
        case DType::INT32:
        case DType::UINT32:
        case DType::FLOAT32:    return 4;
        case DType::INT64:
        case DType::UINT64:
        case DType::FLOAT64:
        case DType::COMPLEX64:  return 8;
        case DType::COMPLEX128: return 16;
        case DType::UNKNOWN:
        default:
            throw IllegalArgumentException("Unknown dtype has no defined size");
    }
}

/** Human-readable label of a dtype (e.g. "float32", "complex64"). */
inline const char *dtypeName(DType dtype) {
    switch (dtype) {
        case DType::UNKNOWN:    return "unknown";
        case DType::BOOL:       return "bool";
        case DType::INT8:       return "int8";
        case DType::UINT8:      return "uint8";
        case DType::INT16:      return "int16";
        case DType::UINT16:     return "uint16";
        case DType::INT32:      return "int32";
        case DType::UINT32:     return "uint32";
        case DType::INT64:      return "int64";
        case DType::UINT64:     return "uint64";
        case DType::FLOAT32:    return "float32";
        case DType::FLOAT64:    return "float64";
        case DType::COMPLEX64:  return "complex64";
        case DType::COMPLEX128: return "complex128";
    }
    return "invalid";
}

/** True iff dtype represents complex numbers (complex64, complex128). */
inline bool isComplex(DType dtype) {
    return dtype == DType::COMPLEX64 || dtype == DType::COMPLEX128;
}

/** True iff dtype represents a floating-point value (float32, float64, complex*). */
inline bool isFloating(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32:
        case DType::FLOAT64:
        case DType::COMPLEX64:
        case DType::COMPLEX128:
            return true;
        default:
            return false;
    }
}

} // namespace arrus::framework

#endif // ARRUS_CORE_API_FRAMEWORK_DTYPE_H
