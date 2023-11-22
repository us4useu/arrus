#ifndef ARRUS_COMMON_CACHE_H
#define ARRUS_COMMON_CACHE_H

#include <optional>

namespace arrus {

/**
 * A single-value cache, with lazy init.
 */
template<typename T>
class Cached {
public:
    explicit Cached(std::function<T()> getter): getter(getter) {}

    T &get() {
        if(!value.has_value()) {
            value = getter();
        }
        return value.value();
    }

private:
    std::function<T()> getter{};
    std::optional<T> value{std::nullopt};
};

}

#endif//ARRUS_COMMON_CACHE_H
