#ifndef ARRUS_CORE_COMMON_EXCEPTIONS_H
#define ARRUS_CORE_COMMON_EXCEPTIONS_H

#include <stdexcept>

namespace arrus {

class ArrusException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class IllegalArgumentException : public ArrusException {
    using ArrusException::ArrusException;
};


}

#endif //ARRUS_CORE_COMMON_EXCEPTIONS_H
