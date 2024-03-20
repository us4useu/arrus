#ifndef ARRUS_CORE_API_COMMON_TUPLE_H
#define ARRUS_CORE_API_COMMON_TUPLE_H

#include <vector>
#include <numeric>
#include <ostream>

namespace arrus {

/**
 * A tuple of values.
 *
 * Note: this class is immutable.
 */
template<typename T>
class Tuple {
public:
    Tuple() = default;

    Tuple(const std::initializer_list<T> &values) : values(values) {}

    explicit Tuple(const std::vector<T> &values) : values(values) {}

    /**
     * Returns i-th. value.
     */
    const T &operator[](size_t i) const {
        return values[i];
    }

    /**
     * Returns i-th value.
     */
    const T &get(size_t i) const {
        return this->values[i];
    }

    /**
     * Returns i-th value.
     */
    T &getMutable(size_t i) {
        return this->values[i];
    }

    Tuple<T> set(size_t i, T value) const {
        std::vector<T> newValues(values);
        newValues[i] = value;
        return Tuple{newValues};
    }

    /**
     * Returns the tuple size (number of values it consists of).
     */
    size_t size() const {
        return values.size();
    }

    const std::vector<T> &getValues() const {
        return values;
    }

    size_t product() const {
        size_t result = 1;
        for(auto &value: values) {
            result = result * value;
        }
        return result;
    }

    size_t sum() const {
        size_t result = 0;
        for(auto &value: values) {
            result = result + value;
        }
        return result;
    }

    bool operator==(const Tuple &rhs) const {
        return values == rhs.values;
    }

    bool operator!=(const Tuple &rhs) const {
        return !(rhs == *this);
    }

    bool empty() const {
        return values.empty();
    }

private:
    std::vector<T> values;
};

}

#endif //ARRUS_CORE_API_COMMON_TUPLE_H
