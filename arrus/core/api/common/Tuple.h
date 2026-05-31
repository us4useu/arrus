#ifndef ARRUS_CORE_API_COMMON_TUPLE_H
#define ARRUS_CORE_API_COMMON_TUPLE_H

#include <initializer_list>
#include <utility>
#include <vector>

#include "arrus/core/api/common/exceptions.h"
#include "std4us/StdInterop.hpp"
#include "std4us/Vector.hpp"

namespace arrus {

/**
 * A tuple of values.
 *
 * Note: this class is immutable.
 *
 * Storage uses std4us::Vector for ABI stability across MSVC Debug/Release
 * builds. The std::vector / std::initializer_list constructors and the
 * getValues() accessor are kept as inline header-only backward-compatibility
 * shims — they never cross the library boundary as std types.
 */
template<typename T>
class Tuple {
public:
    Tuple() = default;

    Tuple(std::initializer_list<T> values) {
        this->values.reserve(values.size());
        for (const auto &v : values) {
            this->values.pushBack(v);
        }
    }

    explicit Tuple(const std::vector<T> &values) : values(std4us::fromStd(values)) {}

    explicit Tuple(std4us::Vector<T> values) : values(std::move(values)) {}

    const T &operator[](size_t i) const { return values[i]; }

    const T &get(size_t i) const { return this->values[i]; }

    T &getMutable(size_t i) { return this->values[i]; }

    Tuple<T> set(size_t i, T value) const {
        std4us::Vector<T> newValues(values);
        newValues[i] = std::move(value);
        return Tuple{std::move(newValues)};
    }

    size_t size() const { return values.size(); }

    /**
     * Backward-compatibility accessor returning a std::vector copy of the
     * underlying storage. The copy is constructed in the caller's TU, so the
     * std::vector lifetime stays inside one CRT.
     */
    std::vector<T> getValues() const { return std4us::toStd(values); }

    /**
     * Native accessor exposing the underlying std4us::Vector by reference.
     * Prefer this over getValues() in new code — no copy, ABI-stable.
     */
    const std4us::Vector<T> &getValuesNative() const { return values; }

    size_t product() const {
        size_t result = 1;
        for (const auto &value : values) {
            result = result * value;
        }
        return result;
    }

    size_t sum() const {
        size_t result = 0;
        for (const auto &value : values) {
            result = result + value;
        }
        return result;
    }

    bool operator==(const Tuple &rhs) const { return values == rhs.values; }

    bool operator!=(const Tuple &rhs) const { return !(rhs == *this); }

    bool empty() const { return values.empty(); }

    std::pair<T, T> asPair() const {
        if (values.size() != 2) {
            throw IllegalArgumentException("The tuple must be a pair, "
                                           "actual number of elements: "
                                           + std::to_string(values.size()));
        }
        return std::make_pair(values[0], values[1]);
    }

private:
    std4us::Vector<T> values;
};

}// namespace arrus

#endif//ARRUS_CORE_API_COMMON_TUPLE_H
