#ifndef ARRUS_COMMON_UTILS_H
#define ARRUS_COMMON_UTILS_H

#include <format>
#include <string>
#include <boost/preprocessor.hpp>

#include "arrus/common/asserts.h"

namespace arrus {

template<typename V, typename T>
V safeCast(const T &in, const std::string& paramName,
           const std::string& requiredTypeName) {
    ARRUS_REQUIRES_DATA_TYPE_E(
        in, V, std::runtime_error(std::format("Data type mismatch: value '{}' cannot be safely casted to type {}.",
                                                  paramName, requiredTypeName)));

    return static_cast<V>(in);
}

#define ARRUS_SAFE_CAST(value, dtype) \
    ::arrus::safeCast<dtype>((value), #value, #dtype)

// Based on the answer:
// https://stackoverflow.com/questions/5093460/how-to-convert-an-enum-type-variable-to-a-string#answer-5094430

#define ARRUS_DEFINE_ENUM_WITH_TO_STRING_CASE(r, data, elem)    \
    case data::elem : return BOOST_PP_STRINGIZE(elem);

#define ARRUS_DEFINE_ENUM_WITH_TO_STRING(name, enumerators)                   \
    enum class name {                                                         \
        BOOST_PP_SEQ_ENUM(enumerators)                                        \
    };                                                                        \
                                                                              \
    inline const char* toString(name v) {                                     \
        switch (v) {                                                          \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                ARRUS_DEFINE_ENUM_WITH_TO_STRING_CASE,                        \
                name,                                                         \
                enumerators                                                   \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }

#define ARRUS_DEFINE_ENUM_TO_STRING(name, enumerators)                        \
    inline const char* toString(name v) {                                     \
        switch (v) {                                                          \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                ARRUS_DEFINE_ENUM_WITH_TO_STRING_CASE,                        \
                name,                                                         \
                enumerators                                                   \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }

}

#define ARRUS_ACTIVE_WAIT_TIMEOUT_P(condition, timeout, timeoutException, pause) \
do {                                                                       \
    auto timeoutStart = std::chrono::high_resolution_clock::now();         \
    while (!(condition)) {                                                 \
        if (std::chrono::high_resolution_clock::now() - timeoutStart > (timeout)) { throw (timeoutException); } \
        std::this_thread::sleep_for(std::chrono::milliseconds(pause));         \
    }                                                                      \
} while(0)

#endif //ARRUS_COMMON_UTILS_H
