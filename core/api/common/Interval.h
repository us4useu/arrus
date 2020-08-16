#ifndef ARRUS_CORE_API_COMMON_INTERVAL_H
#define ARRUS_CORE_API_COMMON_INTERVAL_H

#include <utility>

namespace arrus {

template<typename T>
class Interval {

public:
    explicit Interval(const std::pair<T, T> &interval) : interval(interval) {}

    const T &left() const { return interval.first; }

    const T &right() const {return interval.second; }

private:
    std::pair<T, T> interval;
};

}

#endif //ARRUS_CORE_API_COMMON_INTERVAL_H
