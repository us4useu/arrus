#include <iostream>
#include <stdexcept>

#include <boost/core/null_deleter.hpp>
#include <boost/log/core.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>

#include "arrus/common/logging/LogSeverity.h"
#include "arrus/common/logging/impl/Logging.h"


BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", arrus::LogSeverity)
BOOST_LOG_ATTRIBUTE_KEYWORD(deviceIdLogAttr, "DeviceId", std::string)

namespace arrus {

static void
addTextSinkBoostPtr(const boost::shared_ptr<std::ostream> &ostream,
            LogSeverity minSeverity) {
    typedef boost::log::sinks::synchronous_sink<
            boost::log::sinks::text_ostream_backend> textSink;
    boost::shared_ptr<textSink> sink = boost::make_shared<textSink>();

    sink->locked_backend()->add_stream(ostream);
    sink->set_filter(severity >= minSeverity);

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

Logging::Logging() {
    boost::log::add_common_attributes();
}

void
Logging::addTextSink(std::shared_ptr<std::ostream> &ostream,
                     LogSeverity minSeverity) {
    boost::shared_ptr<std::ostream> boostPtr = boost::shared_ptr<std::ostream>(
            ostream.get(),
            [ostream](std::ostream *) mutable {ostream.reset();});
    addTextSinkBoostPtr(boostPtr, minSeverity);
}

void Logging::addClog(LogSeverity severity) {
    boost::shared_ptr<std::ostream> stream(&std::clog, boost::null_deleter());
    addTextSinkBoostPtr(stream, severity);
}

Logger::Handle Logging::getLogger() {
    return std::make_unique<LoggerImpl>();
}

Logger::Handle
Logging::getLogger(const std::vector<arrus::Logger::Attribute> &attributes) {
    return std::make_unique<LoggerImpl>(attributes);
}


}
