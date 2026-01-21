#ifndef ARRUS_CORE_COMMON_MATH_H
#define ARRUS_CORE_COMMON_MATH_H
#include <cmath>

namespace arrus {

/**
 * Rounds the given value to the given number of decimal places.
 */
template<typename T>
T roundTo(T value, int decimalPlaces) {
    double factor = std::pow(10.0, decimalPlaces);
    return std::round(value * factor) / factor;
}

bool almostEqual(float a, float b, float epsilon = 1e-5f) {
    return (a < b) || (std::fabs(a - b) < epsilon);
}

}



#endif//ARRUS_CORE_COMMON_MATH_H
