#ifndef ARRUS_CORE_UTILS_ASSERTS_H
#define ARRUS_CORE_UTILS_ASSERTS_H

#include "arrus/core/api/common/exceptions.h"

#define ARRUS_REQUIRES_TRUE(CONDITION, MSG) \
do {                                        \
    if (!(CONDITION)) {                     \
        throw arrus::ArrusException((MSG)); \
    }                                       \
} while(0)

#define ARRUS_REQUIRES_EQUAL(A, B, EXCEPTION)             \
do {                                                      \
    if (!((A) == (B))) {                                  \
        throw EXCEPTION;                                  \
    }                                                     \
} while(0)

#define ARRUS_REQUIRES_TRUE_FOR_ARGUMENT(CONDITION, MSG) \
do {                                                     \
    if (!(CONDITION)) {                                  \
        throw arrus::IllegalArgumentException((MSG));    \
    }                                                    \
} while(0)

#define ARRUS_REQUIRES_NO_THROW(EXPR, IN_EXCEPTION_TYPE, OUT_EXCEPTION) \
do {                                                                    \
    try {                                                               \
        EXPR;                                                           \
    } catch(const IN_EXCEPTION_TYPE &e) {                               \
        throw (OUT_EXCEPTION);                                          \
    }                                                                   \
} while(0)

/**
 * Check if A >= B, otherwise throws arrus::IllegalArgumentException.
 */
#define ARRUS_REQUIRES_AT_LEAST(A, B, MSG)                \
do {                                                      \
    if (!((A) >= (B))) {                                  \
        throw arrus::IllegalArgumentException((MSG));     \
    }                                                     \
} while(0)

/**
 * Check if A >= B, otherwise throws arrus::IllegalArgumentException.
 */
#define ARRUS_REQUIRES_AT_MOST(A, B, MSG)                \
do {                                                      \
    if (!((A) <= (B))) {                                  \
        throw arrus::IllegalArgumentException((MSG));     \
    }                                                     \
} while(0)


#endif //ARRUS_CORE_UTILS_ASSERTS_H
