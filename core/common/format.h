#ifndef ARRUS_CORE_COMMON_FORMAT_H
#define ARRUS_CORE_COMMON_FORMAT_H

// String formatting utilities.
// Currently wraps fmt library calls.
#include <fmt/format.h>

namespace arrus {

template <typename... Args>
auto format(Args&&... args) {
    return fmt::format(std::forward<Args>(args)...);
}

}

#endif //ARRUS_CORE_COMMON_FORMAT_H
