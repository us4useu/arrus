#ifndef ARRUS_COMMON_LOGGING_IMPL_LOGGERIMPL_H
#define ARRUS_COMMON_LOGGING_IMPL_LOGGERIMPL_H

#include <boost/log/sources/severity_feature.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/core.hpp>
#include <boost/log/attributes.hpp>

#include "arrus/core/api/common/Logger.h"

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

    LoggerImpl() = default;

    /**
     * Creates a logger with DeviceId attribute set.
     *
     * @param attributes attributes to set
     */
    explicit LoggerImpl(const std::vector<Logger::Attribute> &attributes) {
        for(auto &[key, value] : attributes) {
            logger.add_attribute(key, boost::log::attributes::constant<std::string>(value));
        }
    }

    /**
     * Logs a given string message with given severity level.
     *
     * @param severity severity attached to the message
     * @param msg message to log
     */
    void log(const LogSeverity severity, const std::string &msg) override {
        BOOST_LOG_SEV(logger, severity) << msg;
    }

    void setAttribute(const std::string& key, const std::string& value) {
        auto attrs_map = logger.get_attributes();
        auto it = attrs_map.find(key);
        if (it != attrs_map.end()) {
            // try to cast to mutable attribute
            auto attr = it->second;
            auto mutable_attr = boost::log::attribute_cast<boost::log::attributes::mutable_constant<std::string>>(attr);
            if (mutable_attr) {
                // Mutable -- change the value.
                mutable_attr.set(value);
            }
            // Otherwise: do nothing: non-mutable attributes are currently not supported.
        }
        // Set the mutable attribute.
        logger.add_attribute(
            key,
            boost::log::attributes::mutable_constant<std::string>(value)
        );
    }

private:
    boost::log::sources::severity_logger_mt<LogSeverity> logger;
};
}

#endif //ARRUS_COMMON_LOGGING_IMPL_LOGGERIMPL_H
