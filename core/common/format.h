#ifndef ARRUS_CORE_COMMON_FORMAT_H
#define ARRUS_CORE_COMMON_FORMAT_H

// String formatting and parsing utilities.
// Currently wraps fmt library calls.
#include <fmt/format.h>
#include <stdexcept>

namespace arrus {

template <typename... Args>
auto format(Args&&... args) {
    return fmt::format(std::forward<Args>(args)...);
}

inline unsigned int stoui(const std::string& str) {
    unsigned long ul = std::stoul(str);
    unsigned int ui = (unsigned int) ul;
    if (ui != ul) {
        throw std::out_of_range("Value out of uint32 range: " + str);
    }
    return ui;
}

}

#endif //ARRUS_CORE_COMMON_FORMAT_H
