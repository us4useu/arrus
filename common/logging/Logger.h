#ifndef ARRUS_COMMON_LOGGING_LOGGER_H
#define ARRUS_COMMON_LOGGING_LOGGER_H

#include <memory>
#include "arrus/common/logging/LogSeverity.h"

namespace arrus {

/**
 * Basic logger instance that can be used in the arrus library.
 *
 * Currently, it is a simple wrapper over boost::severity_logger_mt.
 *
 * This class should not be available publicly.
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
};

}

#endif //ARRUS_COMMON_LOGGING_LOGGER_H
