#ifndef ARRUS_COMMON_ASSERTS_H
#define ARRUS_COMMON_ASSERTS_H

#include "arrus/core/api/common/exceptions.h"

#define ARRUS_REQUIRES_TRUE(CONDITION, MSG) \
do {                                        \
    if (!(CONDITION)) {                     \
        throw ::arrus::ArrusException((MSG)); \
    }                                       \
} while(0)


#define ARRUS_REQUIRES_TRUE_E(CONDITION, EXCEPTION) \
do {                                        \
    if (!(CONDITION)) {                     \
        throw (EXCEPTION); \
    }                                       \
} while(0)

#define ARRUS_REQUIRES_EQUAL(A, B, EXCEPTION)             \
do {                                                      \
    if (!((A) == (B))) {                                  \
        throw (EXCEPTION);                                  \
    }                                                     \
} while(0)

#define ARRUS_REQUIRES_EQUAL_IAE(A, B) \
    ARRUS_REQUIRES_EQUAL(A, B, IllegalArgumentException(#A " != " #B))

#define ARRUS_REQUIRES_NON_EMPTY_IAE(coll)             \
do {                                                      \
    if ((coll).empty()) {                                  \
        throw IllegalArgumentException(#coll " cannot be empty"); \
    }                                                     \
} while(0)

#define ARRUS_REQUIRES_TRUE_FOR_ARGUMENT(CONDITION, MSG) \
do {                                                     \
    if (!(CONDITION)) {                                  \
        throw ::arrus::IllegalArgumentException((MSG));    \
    }                                                    \
} while(0)

#define ARRUS_REQUIRES_NO_THROW(EXPR, IN_EXCEPTION_TYPE, OUT_EXCEPTION) \
do {                                                                    \
    try {                                                               \
        EXPR;                                                           \
    } catch(const IN_EXCEPTION_TYPE&) {                               \
        throw (OUT_EXCEPTION);                                          \
    }                                                                   \
} while(0)

/**
 * Check if A >= B, otherwise throws arrus::IllegalArgumentException.
 */
#define ARRUS_REQUIRES_AT_LEAST(A, B, MSG)                \
do {                                                      \
    if (!((A) >= (B))) {                                  \
        throw ::arrus::IllegalArgumentException((MSG));     \
    }                                                     \
} while(0)

/**
 * Check if A >= B, otherwise throws arrus::IllegalArgumentException.
 */
#define ARRUS_REQUIRES_AT_MOST(A, B, MSG)                \
do {                                                      \
    if (!((A) <= (B))) {                                  \
        throw ::arrus::IllegalArgumentException((MSG));     \
    }                                                     \
} while(0)

/**
 * Check if value in range [min, max], otherwise throws arrus::IllegalArgumentException.
 */
#define ARRUS_REQUIRES_IN_CLOSED_INTERVAL(value, min, max, MSG) \
do {                                                      \
    if (!(((value) >= (min) && (value) <= (max)))) {      \
        throw ::arrus::IllegalArgumentException((MSG));   \
    }                                                     \
} while(0)

#define ARRUS_REQUIRES_IN_CLOSED_INTERVAL_E(value, min, max, exception) \
do {                                                      \
    if (!(((value) >= (min) && (value) <= (max)))) {      \
        throw (exception);                                  \
    }                                                     \
} while(0)

#define ARRUS_REQUIRES_DATA_TYPE_E(value, dtype, exception) \
    ARRUS_REQUIRES_IN_CLOSED_INTERVAL_E(value,        \
        (std::numeric_limits<dtype>::min)(),   \
        (std::numeric_limits<dtype>::max)(),   \
        exception)

#define ARRUS_REQUIRES_DATA_TYPE(value, dtype, msg) \
     ARRUS_REQUIRES_DATA_TYPE_E(value, dtype, ::arrus::IllegalArgumentException(msg)) \

#define ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(cv, lock, timeout, exceptionMsg) \
    if((timeout) > -1) { \
        auto status = (cv).wait_for(lock ,std::chrono::milliseconds(timeout)); \
        if(status == std::cv_status::timeout) { \
            throw TimeoutException(exceptionMsg); \
        } \
    } \
    else { \
        (cv).wait(lock); \
    }
#endif //ARRUS_COMMON_ASSERTS_H
