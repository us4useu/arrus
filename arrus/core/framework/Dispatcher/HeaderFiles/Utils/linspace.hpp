/*
* linspace.hpp
*
*  Created on: Jun 23, 2015
*      Author: marcin
*/

#ifndef LINSPACE_HPP_
#define LINSPACE_HPP_

#include <vector>

template<typename T>
std::vector <T> linspace(const T lo, const T hi, const int n) {
    T incr = (hi - lo) / (n - 1);
    std::vector <T> res(n);

    for(int i = 0; i < n; ++i) {
        res[i] = lo + i * incr;
    }

    return res;

}

#endif /* LINSPACE_HPP_ */
