#ifndef ARRUS_CORE_COMMON_EXCEPTIONS_H
#define ARRUS_CORE_COMMON_EXCEPTIONS_H

#include <stdexcept>
#include "arrus/core/api/devices/DeviceId.h"

namespace arrus {

class ArrusException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class IllegalArgumentException : public ArrusException {
public:
    using ArrusException::ArrusException;
};

class DeviceNotFoundException : public IllegalArgumentException {
public:
    explicit DeviceNotFoundException(const DeviceId &id)
        : IllegalArgumentException("Device " + id.toString() + " not found.") {}
};


}

#endif //ARRUS_CORE_COMMON_EXCEPTIONS_H
