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

    explicit Tuple(const std::vector<T> &values) : values(values) {}

    const T &operator[](size_t i) const {
        return values[i];
    }

    const T &get(size_t i) const {
        return this->values[i];
    }

    size_t size() const {
        return values.size();
    }

    const std::vector<T> &getValues() const {
        return values;
    }

    size_t product() const {
        return std::reduce(
                std::begin(values), std::end(values), size_t(1),
                [](auto v1, auto v2) -> size_t {
                	return v1 * v2;
                }
        );
    }

    size_t sum() const {
        return std::reduce(
            std::begin(values), std::end(values), size_t(0),
            [](auto v1, auto v2) -> size_t {
                return v1 + v2;
            }
        );
    }

    bool operator==(const Tuple &rhs) const {
        return values == rhs.values;
    }

    bool operator!=(const Tuple &rhs) const {
        return !(rhs == *this);
    }

private:
    std::vector<T> values;
};

}

#endif //ARRUS_CORE_API_COMMON_TUPLE_H
