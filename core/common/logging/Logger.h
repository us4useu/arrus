#ifndef ARRUS_CORE_COMMON_LOGGING_LOGGER_H
#define ARRUS_CORE_COMMON_LOGGING_LOGGER_H

#include <boost/log/sources/severity_feature.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/core.hpp>
#include <boost/log/attributes.hpp>

#include "LogSeverity.h"
#include "LogSettings.h"
#include "core/common/format.h"

#include "core/devices/DeviceId.h"

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

    /**
     * Returns a default instance of the logger.
     *
     * @return a default instance of the logger
     */
    static Logger &get() {
        static Logger instance;
        return instance;
    }

    Logger() {
        // Instantiate configuration to make sure that at least the default
        // settings are set.
        LogSettings::getInstance();
    }

    /**
     * Creates a logger with DeviceId attribute set.
     *
     * @param deviceId device id to set to given logger.
     */
    explicit Logger(const DeviceId &deviceId): Logger() {
        logger.add_attribute("DeviceId",
                             boost::log::attributes::constant<std::string>(
                                     deviceId.toString()));
    }

    /**
     * Logs a given string message with given severity level.
     *
     * @param severity severity attached to the message
     * @param msg message to log
     */
    void log(const LogSeverity severity, const std::string &msg) {
        BOOST_LOG_SEV(logger, severity) << msg;
    }

    /**
     * Logs a given message with the given severity level.
     *
     * The msg string can be a arrus::format input; the following parameters
     * will be treated as an input parameters to generate the final string
     * message.
     *
     * @param severity severity attached to the message
     * @param msg message to log
     * @param args values to put in the log message
     */
    template<typename... Args>
    void logFmt(const LogSeverity severity, const std::string &msg,
                Args &&... args) {
        BOOST_LOG_SEV(logger, severity) << arrus::format(msg, args...);
    }

private:
    boost::log::sources::severity_logger_mt<LogSeverity> logger;
};
}

#endif //ARRUS_CORE_COMMON_LOGGING_LOGGER_H
