#ifndef ARRUS_COMMON_LOGGING_IMPL_LOGGERIMPL_H
#define ARRUS_COMMON_LOGGING_IMPL_LOGGERIMPL_H

#include <boost/log/sources/severity_feature.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/core.hpp>
#include <boost/log/attributes.hpp>

#include "arrus/common/logging/Logger.h"

namespace arrus {

/**
 * Basic logger instance that can be used in the arrus library.
 *
 * Currently, it is a simple wrapper over boost::severity_logger_mt.
 *
 * This class should not be available publicly.
 */
class LoggerImpl : public Logger {
public:
    /**
     * Creates a logger with DeviceId attribute set.
     *
     * @param attributes attributes to set
     */
    explicit LoggerImpl(const std::vector<Logger::Attribute> &attributes) {
        for(auto &[key, value] : attributes) {
            logger.add_attribute(key,
                                 boost::log::attributes::constant<std::string>(
                                         value));
        }
    }

    /**
     * Logs a given string message with given severity level.
     *
     * @param severity severity attached to the message
     * @param msg message to log
     */
    void
    log(const LogSeverity severity, const std::string &msg) override {
        BOOST_LOG_SEV(logger, severity) << msg;
    }

private:
    boost::log::sources::severity_logger_mt<LogSeverity> logger;
};
}

#endif //ARRUS_COMMON_LOGGING_IMPL_LOGGERIMPL_H
