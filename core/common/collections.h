#ifndef ARRUS_CORE_COMMON_COLLECTIONS_H
#define ARRUS_CORE_COMMON_COLLECTIONS_H

#include <string>
#include <vector>
#include <numeric>
#include <unordered_set>

namespace arrus {

/**
 * Returns an array of range [start, end).
 */
template<typename T>
inline std::vector<T> getRange(T start, T end, T step = 1) {
    std::vector<T> values;
    for(T i = start; i < end; i += step) {
        values.push_back(i);
    }
    return values;
}

/**
 * Returns an array that holds given value n times.
 */
template<typename T>
inline std::vector<T> getNTimes(const T value, size_t n) {
    std::vector<T> values;
    for(size_t i = 0; i < n; ++i) {
        values.push_back(value);
    }
    return values;
}

template<typename T>
inline size_t countUnique(const std::vector<T> values) {
    return std::unordered_set<T>(std::begin(values), std::end(values)).size();
}

}

#endif //ARRUS_CORE_COMMON_COLLECTIONS_H
