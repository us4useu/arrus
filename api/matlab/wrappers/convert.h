#ifndef ARRUS_API_MATLAB_WRAPPERS_CONVERT_H
#define ARRUS_API_MATLAB_WRAPPERS_CONVERT_H

#include "mex.hpp"
#include "mexAdapter.hpp"
// A header with converting routines mex - cpp for basic types: string, matlab,
// etc.

namespace arrus::matlab {

    std::string convertToString(const ::matlab::data::CharArray& charArray) {
        return charArray.toAscii();
    }

}


#endif //ARRUS_API_MATLAB_WRAPPERS_CONVERT_H
