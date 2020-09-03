#ifndef ARRUS_CORE_COMMON_COLLECTIONS_H
#define ARRUS_CORE_COMMON_COLLECTIONS_H

#include <string>
#include <vector>
#include <numeric>
#include <unordered_set>
#include <type_traits>

#include <range/v3/all.hpp>

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

template<typename In, typename Out>
inline std::vector<Out> castTo(std::vector<In> values) {
    std::vector<Out> result(values.size());
    std::transform(
            std::begin(values), std::end(values),
            std::begin(result),
            [] (In &value) {return Out(value);}
   );
   return result;
}

template<typename T>
inline std::vector<T> toVector(ranges::iota_view<T> values) {
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
inline size_t countUnique(const std::vector<T> &values) {
    return std::unordered_set<T>(std::begin(values), std::end(values)).size();
}

template<typename T>
inline bool
setContains(const std::unordered_set<T> &set, const T &value) {
    return set.find(value) != set.end();
}

template<typename T, typename U>
inline std::vector<std::pair<T, U>>
// TODO ranges as input might more efficient here
zip(const std::vector<T> &a, const std::vector<U> &b) {
    std::vector<std::pair<T, U>> res = ranges::view::zip(a, b);
    return res;
}

template<typename R>
inline std::vector<R>
generate(size_t nElements, std::function<R(size_t)> transformation) {
    std::vector<R> result;
    for(auto i : ranges::view::ints((size_t) 0, nElements)) {
        result.emplace_back(transformation(i));
    }
    return result;
}

template<typename T>
inline std::vector<T>
concat(const std::vector<T> &a, const std::vector<T> &b) {
    std::vector<T> result;
    result.reserve(a.size() + b.size());
    result.insert(std::begin(result), std::begin(a), std::end(a));
    result.insert(std::end(result), std::begin(b), std::end(b));
    return result;

}

template<typename T>
inline std::vector<T>
concat(const std::vector<std::vector<T>> &a) {
    std::vector<T> result;
    size_t totalSize = 0;
    for(const auto &v : a) {
        totalSize += v.size();
    }
    result.reserve(totalSize);
    for(const auto &vec : a) {
        result.insert(std::end(result), std::begin(vec), std::end(vec));
    }
    return result;
}

}

#endif //ARRUS_CORE_COMMON_COLLECTIONS_H
