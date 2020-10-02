#ifndef ARRUS_CORE_COMMON_COLLECTIONS_H
#define ARRUS_CORE_COMMON_COLLECTIONS_H

#include <string>
#include <vector>
#include <numeric>
#include <unordered_set>
#include <type_traits>
#include <bitset>

#include <gsl/span>
#include <boost/range/combine.hpp>

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

template<typename Out, typename In>
inline std::vector<Out> castTo(std::vector<In> values) {
    std::vector<Out> result(values.size());
    std::transform(
        std::begin(values), std::end(values),
        std::begin(result),
        [](In &value) { return Out(value); }
    );
    return result;
}

template<typename Out, typename Iterator>
inline std::vector<Out> castTo(const Iterator begin, const Iterator end) {
    std::vector<Out> result;
    std::transform(begin, end, std::back_inserter(result),
                   [](auto &value) { return Out(value); });
    return result;
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
zip(const std::vector<T> &a, const std::vector<U> &b) {
    if(a.size() != b.size()) {
        throw IllegalArgumentException("Zipped vectors should "
                                       "have the same size.");
    }
    std::vector<std::pair<T, U>> res;
    res.reserve(a.size());
    for(const auto &[x, y] : boost::combine(a, b)) {
        res.emplace_back(x, y);
    }
    return res;
}

template<typename R>
inline std::vector<R>
generate(size_t nElements, std::function<R(size_t)> transformation) {
    std::vector<R> result;
    result.reserve(nElements);
    for(size_t i = 0; i < nElements; ++i) {
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

template<typename T>
inline std::vector<T>
permute(const std::vector<T> &input, const std::vector<ChannelIdx> &perm) {
    std::vector<T> output(perm.size());
    for(size_t i = 0; i < static_cast<size_t>(perm.size()); ++i) {
        output[i] = input[perm[i]];
    }
    return output;
}

template<int size>
inline std::bitset<size>
toBitset(const std::vector<bool> &in) {
    std::bitset<size> result;
    for(size_t i = 0; i < size; ++i) {
        result[i] = in[i];
    }
    return result;
}

template<typename Map, typename K>
inline bool containsKey(Map map, const K &key) {
    return map.find(key) != std::end(map);
}

template<typename T>
inline void setValuesInRange(std::vector<T> &container, size_t start, size_t end, const T &value) {
    for(size_t i = start; i < end; ++i) {
        container[i] = value;
    }
}

template<typename T>
inline void setValuesInRange(std::vector<T> &container, size_t start, size_t end,
                             const std::function<T(size_t)> &generator) {
    for(size_t i = start; i < end; ++i) {
        container[i] = generator(i);
    }
}

template<typename T>
inline void setValuesInRange(std::vector<T> &container, size_t start, size_t end,
                             const std::vector<T>& source) {
    if(source.size != end-start) {
        throw IllegalArgumentException(
            arrus::format("Source vector should have exactly {} elements "
                          "when assing it to the selected range.",
                          end-start));
    }
    for(size_t i = start; i < end; ++i) {
        container[i] = source[i-start];
    }
}

}

#endif //ARRUS_CORE_COMMON_COLLECTIONS_H
