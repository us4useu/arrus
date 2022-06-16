#ifndef CPP_EXAMPLE_IMAGING_UTILS_H
#define CPP_EXAMPLE_IMAGING_UTILS_H

#include <vector>

std::vector<float> arange(float l, float r, float step) {
    std::vector<float> result;
    float currentPos = l;
    while (currentPos <= r) {
        result.push_back(currentPos);
        currentPos += step;
    }
    return result;
}


#endif//CPP_EXAMPLE_IMAGING_UTILS_H
