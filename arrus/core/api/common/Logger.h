#ifndef ARRUS_CORE_API_COMMON_LOGGER_H
#define ARRUS_CORE_API_COMMON_LOGGER_H

#include <format>
#include <memory>
#include "LogSeverity.h"

#include <std4us/concepts.h>
#include <std4us/string.h>

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

#ifndef SWIG // Swig cannot generate code for variadic templates.

    /**
     * Logs a formatted message (using std::format as the backend) with given severity level.
     *
     * @param severity severity attached to the message
     * @param fmt format string
     * @param args arguments for the format string - see std::format
     */
    template<typename... Args> requires (std::formattable<Args, char> && ...)
    void log(const LogSeverity severity, const std::string &fmt, Args... args) {
        log(severity, std::vformat(fmt, std::make_format_args(args...)));
    }

    template<typename... Args> requires ((!std::formattable<Args, char> && std4us::supports_to_string<Args>) && ...)
    void log(const LogSeverity severity, const std::string &fmt, Args... args) {
        log(severity, std::vformat(fmt, std::make_format_args(std4us::to_string(args)...)));
    }

    /**
     * Logs a formatted message (using std::format as the backend) at trace level.
     * 
     * @param fmt format string
     * @param args arguments for the format string - see std::format
     */
    template<typename... Args>
    void trace(const std::string &fmt, Args... args) {
        log(LogSeverity::TRACE, fmt, args...);
    }

    /**
     * Logs a formatted message (using std::format as the backend) at debug level.
     * 
     * @param fmt format string
     * @param args arguments for the format string - see std::format
     */
    template<typename... Args>
    void debug(const std::string &fmt, Args... args) {
        log(LogSeverity::DEBUG, fmt, args...);
    }

    /**
     * Logs a formatted message (using std::format as the backend) at info level.
     * 
     * @param fmt format string
     * @param args arguments for the format string - see std::format
     */
    template<typename... Args>
    void info(const std::string &fmt, Args... args) {
        log(LogSeverity::INFO, fmt, args...);
    }

    /**
     * Logs a formatted message (using std::format as the backend) at warning level.
     * 
     * @param fmt format string
     * @param args arguments for the format string - see std::format
     */
    template<typename... Args>
    void warn(const std::string &fmt, Args... args) {
        log(LogSeverity::WARNING, fmt, args...);
    }

    /**
     * Logs a formatted message (using std::format as the backend) at error level.
     * 
     * @param fmt format string
     * @param args arguments for the format string - see std::format
     */
    template<typename... Args>
    void error(const std::string &fmt, Args... args) {
        log(LogSeverity::ERROR, fmt, args...);
    }

    /**
     * Logs a formatted message (using std::format as the backend) at fatal level.
     * 
     * @param fmt format string
     * @param args arguments for the format string - see std::format
     */
    template<typename... Args> requires (std::formattable<Args, char> && ...)
    void fatal(const std::string &fmt, Args... args) {
        log(LogSeverity::FATAL, fmt, args...);
    }

#endif

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
