#ifndef ARRUS_CORE_API_VERSION_H
#define ARRUS_CORE_API_VERSION_H

#include <string_view>
#include "arrus/core/api/common/macros.h"

namespace arrus {
    /**
     Returns ARRUS version. Format: "major.minor.patch" (optionally-dev[date]).
     */
    ARRUS_CPP_EXPORT
    const char *version();
}




#endif//ARRUS_CORE_API_VERSION_H
