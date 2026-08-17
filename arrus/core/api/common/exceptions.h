#ifndef ARRUS_CORE_COMMON_EXCEPTIONS_H
#define ARRUS_CORE_COMMON_EXCEPTIONS_H

#include <format>
#include <stdexcept>
#include "arrus/core/api/devices/DeviceId.h"

namespace arrus {

class ArrusException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

    template<typename... Args>
    ArrusException(const std::string &fmt, Args... args)
        : std::runtime_error(std::vformat(fmt, std::make_format_args(args...))) {}
};

class IllegalArgumentException : public ArrusException {
public:
    using ArrusException::ArrusException;

    template<typename... Args>
    IllegalArgumentException(const std::string &fmt, Args... args)
        : ArrusException(std::vformat(fmt, std::make_format_args(args...))) {}
};

class DeviceNotFoundException : public IllegalArgumentException {
public:
    explicit DeviceNotFoundException(const arrus::devices::DeviceId &id)
        : IllegalArgumentException("Device {} not found.", id.toString()) {}
};

class IllegalStateException : public ArrusException {
public:
    using ArrusException::ArrusException;

    template<typename... Args>
    IllegalStateException(const std::string &fmt, Args... args)
        : ArrusException(std::vformat(fmt, std::make_format_args(args...))) {}
};

class TimeoutException : public ArrusException {
public:
    using ArrusException::ArrusException;

    template<typename... Args>
    TimeoutException(const std::string &fmt, Args... args)
        : ArrusException(std::vformat(fmt, std::make_format_args(args...))) {}
};


}

#endif //ARRUS_CORE_COMMON_EXCEPTIONS_H
