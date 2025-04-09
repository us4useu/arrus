#ifndef ARRUS_CORE_API_VERSION_H
#define ARRUS_CORE_API_VERSION_H

#include <string_view>

namespace arrus {
    /**
     Returns ARRUS version. Format: "major.minor.patch" (optionally-dev[date]).
     */
     const char *version();
}




#endif//ARRUS_CORE_API_VERSION_H
