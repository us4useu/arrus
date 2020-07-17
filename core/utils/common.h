#ifndef ARRUS_CORE_UTILS_H
#define ARRUS_CORE_UTILS_H

#include <cstddef>

#include <boost/functional/hash.hpp>

namespace arrus {

// Hash
inline void hash_combine_seed(std::size_t &seed) {}

template<typename T, typename... Rest>
inline void hash_combine_seed(std::size_t &seed, const T &v, Rest... rest) {
    boost::hash_combine(seed, v);
    hash_combine_seed(seed, rest...);
}

template<typename T, typename... Rest>
inline std::size_t hash_combine(const T &v, Rest... rest) {
    std::size_t seed = 0;
    hash_combine_seed(seed, v, rest...);
    return seed;
}

#define GET_HASHER_NAME(type) type##Hasher

#define MAKE_HASHER(type, ...) \
    struct GET_HASHER_NAME(type) { \
        std::size_t operator()(const type &t) const { \
            return arrus::hash_combine(__VA_ARGS__); \
        } \
    };

}

#endif //ARRUS_CORE_UTILS_H
