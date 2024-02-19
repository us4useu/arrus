#ifndef ARRUS_CORE_API_COMMON_INTERVAL_H
#define ARRUS_CORE_API_COMMON_INTERVAL_H

#include <utility>

#include "arrus/core/api/common/exceptions.h"

namespace arrus {

template<typename T>
class Interval {
public:
    Interval(): Interval(T(0), T(0)) {}

    Interval(const T &start, const T &end)
            : Interval(std::make_pair(start, end)) {}

    explicit Interval(const std::pair<T, T> &interval) {
        if(interval.first > interval.second) {
            throw IllegalArgumentException("Start should not be greater "
                                           "than the end of the interval.");
        }
        this->interval = {interval.first, interval.second};
    }

    const T &start() const { return interval.first; }

    const T &end() const { return interval.second; }

    std::pair<T, T> asPair() const {return interval;}

    bool operator==(const Interval &rhs) const {
        return interval == rhs.interval;
    }

    bool operator!=(const Interval &rhs) const {
        return !(rhs == *this);
    }

private:
    std::pair<T, T> interval;
};

}

#endif //ARRUS_CORE_API_COMMON_INTERVAL_H
