#ifndef ARRUS_CORE_UTILS_ASSERTS_H
#define ARRUS_CORE_UTILS_ASSERTS_H

#include "core/common/exceptions.h"

#define ARRUS_REQUIRES_TRUE(CONDITION, MSG) \
do {                                        \
    if (!(CONDITION)) {                     \
        throw arrus::ArrusException((MSG)); \
    }                                       \
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


#endif //ARRUS_CORE_UTILS_ASSERTS_H
