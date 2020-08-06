/**
 * This source file initializes boost logger default configuration and attributes.
 */

#include <ostream>
#include <iostream>
#include <stdexcept>


#include <boost/core/null_deleter.hpp>
#include <boost/log/core.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>


#include "arrus/core/devices/DeviceId.h"
#include "arrus/core/common/logging/LogSeverity.h"
#include "arrus/core/common/logging/LogSettings.h"


BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", arrus::LogSeverity);
BOOST_LOG_ATTRIBUTE_KEYWORD(deviceIdLogAttr, "DeviceId", std::string);

namespace arrus {
class LogSettingsImpl {
public:
    typedef boost::shared_ptr<std::ostream> OstreamPtr;

    static void
    addTextSink(const OstreamPtr &ostream, LogSeverity minSeverity) {
        typedef boost::log::sinks::synchronous_sink<
                boost::log::sinks::text_ostream_backend> textSink;
        boost::shared_ptr<textSink> sink = boost::make_shared<textSink>();
        // Add a stream to write log to
        sink->locked_backend()->add_stream(ostream);
        sink->set_filter(severity >= minSeverity);

//        (*ostream) << minSeverity << std::endl;

        namespace expr = boost::log::expressions;

        boost::log::formatter formatter = expr::stream
                << "["
                << expr::format_date_time<boost::posix_time::ptime>(
                        "TimeStamp",
                        "%Y-%m-%d %H:%M:%S")
                << "]"
                << expr::if_(expr::has_attr(deviceIdLogAttr))
                [
                   expr::stream << "[" << deviceIdLogAttr << "]"
                ]
                << " " << severity << ": "
                << expr::smessage;
        sink->set_formatter(formatter);
        boost::log::core::get()->add_sink(sink);
    }
};


LogSettings::LogSettings() {
    boost::log::add_common_attributes();

    boost::shared_ptr<std::ostream> stream(&std::clog, boost::null_deleter());
    LogSettingsImpl::addTextSink(stream, LogSettings::DEFAULT);
}

void
LogSettings::addLogFile(const std::string &filename, LogSeverity severity) {
    throw std::runtime_error("NYI");
}

void LogSettings::setConsoleLogLevel(LogSeverity severity) {
    throw std::runtime_error("NYI");
}


}
