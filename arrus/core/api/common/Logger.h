#ifndef ARRUS_CORE_API_COMMON_LOGGER_H
#define ARRUS_CORE_API_COMMON_LOGGER_H

#include <memory>
#include "LogSeverity.h"

namespace arrus {

/**
 * Basic logger instance that can be used in the arrus library.
 */
class Logger {
public:
    using Handle = std::unique_ptr<Logger>;
    using SharedHandle = std::shared_ptr<Logger>;

    using Attribute = std::pair<std::string, std::string>;

    /**
     * Logs a given string message with given severity level.
     *
     * @param severity severity attached to the message
     * @param msg message to log
     */
    virtual void log(const LogSeverity severity, const std::string &msg) = 0;

    /**
     * Sets logger attribute with given value.
     *
     * This function can be used e.g. to set device id of the device logger.
     *
     * @param key attribute's name
     * @param value value to set
     */
    virtual void
    setAttribute(const std::string &key, const std::string &value) = 0;

    virtual ~Logger() = default;
};

}

#endif //ARRUS_CORE_API_COMMON_LOGGER_H
