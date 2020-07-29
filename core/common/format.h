#ifndef ARRUS_CORE_COMMON_FORMAT_H
#define ARRUS_CORE_COMMON_FORMAT_H

// String formatting and parsing utilities.
// Currently wraps fmt library calls.
#include <fmt/format.h>
#include <stdexcept>
#include <cctype>

namespace arrus {

template <typename... Args>
auto format(Args&&... args) {
    return fmt::format(std::forward<Args>(args)...);
}

/**
 * Returns true if the given string contains numeric characters only.
 *
 * @param num string to verify
 * @return true if the given string contains numeric characters only,
 *         false otherwise.
 */
inline bool isDigitsOnly(const std::string& num) {
    return std::all_of(num.begin(), num.end(), isdigit);
}

}

#endif //ARRUS_CORE_COMMON_FORMAT_H
