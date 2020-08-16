#ifndef ARRUS_CORE_API_COMMON_TUPLE_H
#define ARRUS_CORE_API_COMMON_TUPLE_H

#include <vector>

namespace arrus {

/**
 * A tuple of values.
 */
template<typename T>
class Tuple {
public:
    Tuple(const std::initializer_list<T> &values) : values(values) {}

    const T& operator[](size_t i) const {
        return values[i];
    }

    [[nodiscard]] unsigned long size() const {
        return values.size();
    }

private:
    std::vector<T> values;
};

}

#endif //ARRUS_CORE_API_COMMON_TUPLE_H
