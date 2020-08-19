#ifndef ARRUS_CORE_API_COMMON_TUPLE_H
#define ARRUS_CORE_API_COMMON_TUPLE_H

#include <vector>
#include <numeric>
#include <ostream>

namespace arrus {

/**
 * A tuple of values.
 */
template<typename T>
class Tuple {
public:
    Tuple(const std::initializer_list<T> &values) : values(values) {}

    const T &operator[](size_t i) const {
        return values[i];
    }

    [[nodiscard]] unsigned long size() const {
        return values.size();
    }

    const std::vector<T> &getValues() const {
        return values;
    }

    T product() const {
        return std::reduce(
                std::begin(values), std::end(values), 1,
                [](const auto &v1, const auto &v2) { return v1 * v2; }
        );
    }
private:
    std::vector<T> values;
};

}

#endif //ARRUS_CORE_API_COMMON_TUPLE_H
