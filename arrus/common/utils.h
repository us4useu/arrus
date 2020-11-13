#ifndef ARRUS_COMMON_UTILS_H
#define ARRUS_COMMON_UTILS_H

#include <string>

#include "arrus/common/asserts.h"
#include "arrus/common/format.h"

namespace arrus {

template<typename V, typename T>
V safeCast(const T &in, const std::string& paramName,
           const std::string& requiredTypeName) {
    ARRUS_REQUIRES_DATA_TYPE_E(
        in, V, std::runtime_error(
            ::arrus::format("Data type mismatch: value '{}' cannot be "
                            "safely casted to type {}.",
                            paramName, requiredTypeName)));

    return static_cast<V>(in);
}

#define ARRUS_SAFE_CAST(value, dtype) \
    ::arrus::safeCast<dtype>((value), #value, #dtype)

}

#endif //ARRUS_COMMON_UTILS_H
