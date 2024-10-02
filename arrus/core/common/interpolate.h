#ifndef ARRUS_CORE_COMMON_INTERPOLATE_H
#define ARRUS_CORE_COMMON_INTERPOLATE_H

#include <vector>
#include <algorithm>

#include "arrus/common/format.h"
#include "arrus/common/asserts.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus {

/**
 * Linear interpolation in 1D.
 *
 * @param x a sorted list
 * @param y
 * @param xi a list of interpolated values, may not be sorted
 * @return yi
 */
template<typename T>
std::vector<T> interpolate1d(const std::vector<T> &x, const std::vector<T> &y,
                             const std::vector<T> &xi,
                             const std::optional<T> minFillValue = std::nullopt,
                             const std::optional<T> maxFillValue = std::nullopt
) {
    ARRUS_REQUIRES_TRUE(!x.empty(), "Interpolation 1D: sample points list should not be empty.");
    ARRUS_REQUIRES_TRUE(x.size() == y.size(), "Interpolation 1D: x.size != y.size");
    ARRUS_REQUIRES_TRUE(!xi.empty(), "Interpolation 1D: query points list should not be empty.");

    std::vector<T> result(xi.size());
    int i = 0;
    for(auto value : xi) {
        auto it = std::lower_bound(std::begin(x), std::end(x), value);
        if(it == std::end(x)) {
            // extrapolate, value > element for all elements of x
            if(minFillValue.has_value()) {
                result[i] = minFillValue.value();
            }
            else {
                throw IllegalArgumentException(format(
                    "Interpolation 1D: value {} is out of range [{}, {}].",
                    value, *std::begin(x), *std::prev(std::end(x))));
            }
        } else if(it == std::begin(x)) {
            if(*it == *std::begin(x)) {
                result[i] = y[0];
            } else {
                // Extrapolate value is lower than the first element of x
                if(maxFillValue.has_value()) {
                    result[i] = maxFillValue.value();
                }
                else {
                    throw IllegalArgumentException(format(
                        "Interp 1D: value {} is out of range [{}, {}].",
                        value, *std::begin(x), *std::prev(std::end(x))));
                }
            }
        } else {
            auto pos = std::distance(std::begin(x), it);
            auto x2 = x[pos], y2 = y[pos];
            auto x1 = x[pos-1], y1 = y[pos-1];

            auto slope = (y2-y1)/(x2-x1);
            auto intercept = y1 - slope*x1;
            auto res = slope*value + intercept;
            result[i] = res;
        }
        ++i;
    }
    return result;
}


}

#endif //ARRUS_CORE_COMMON_INTERPOLATE_H
