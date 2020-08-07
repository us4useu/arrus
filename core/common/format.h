#ifndef ARRUS_CORE_COMMON_FORMAT_H
#define ARRUS_CORE_COMMON_FORMAT_H

// String formatting and parsing utilities.
// Currently wraps fmt library calls.
#include <fmt/format.h>
#include <stdexcept>
#include <cctype>
#include <vector>

#include <boost/algorithm/string/join.hpp>

namespace arrus {

template<typename... Args>
auto format(Args &&... args) {
    return fmt::format(std::forward<Args>(args)...);
}

/**
 * Returns true if the given string contains numeric characters only.
 *
 * @param num string to verify
 * @return true if the given string contains numeric characters only,
 *         false otherwise.
 */
inline bool isDigitsOnly(const std::string &num) {
    return std::all_of(num.begin(), num.end(), isdigit);
}

template<typename T>
inline std::string toString(const std::vector<T> values) {
    std::vector<std::string> vStr(values.size());
    std::transform(std::begin(values), std::end(values), std::begin(vStr),
                   [](auto v) {return std::to_string(v);});
    return boost::algorithm::join(vStr, ", ");
}

template<typename T>
inline std::string toString(const std::optional<T> value) {
    if(value.has_value()) {
        return std::to_string(value.value());
    }
    else return "(novalue)";
}
}

#endif //ARRUS_CORE_COMMON_FORMAT_H
