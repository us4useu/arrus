#ifndef ARRUS_CORE_API_COMMON_INTERVAL_H
#define ARRUS_CORE_API_COMMON_INTERVAL_H

#include <utility>

#include "arrus/core/api/common/exceptions.h"

namespace arrus {

template<typename T>
class Interval {

public:

    Interval(const T &left, const T &right)
            : Interval(std::make_pair(left, right)) {}

    explicit Interval(const std::pair<T, T> &interval) {
        if(interval.first > interval.second) {
            throw IllegalArgumentException("Left border should not be greater "
                                           "than the right border.");
        }
        this->interval = {interval.first, interval.second};
    }

    const T &left() const { return interval.first; }

    const T &right() const { return interval.second; }

private:
    std::pair<T, T> interval;
};

}

#endif //ARRUS_CORE_API_COMMON_INTERVAL_H
