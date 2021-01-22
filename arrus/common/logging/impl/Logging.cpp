#include <iostream>
#include <stdexcept>



#include "arrus/core/api/common/LogSeverity.h"
#include "arrus/common/logging/impl/Logging.h"


BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", arrus::LogSeverity)
BOOST_LOG_ATTRIBUTE_KEYWORD(deviceIdLogAttr, "DeviceId", std::string)

namespace arrus {

typedef boost::log::sinks::synchronous_sink<
    boost::log::sinks::text_ostream_backend> textSink;

static boost::shared_ptr<textSink>
addTextSinkBoostPtr(const boost::shared_ptr<std::ostream> &ostream,
                    LogSeverity minSeverity, bool autoFlush) {

    boost::shared_ptr<textSink> sink = boost::make_shared<textSink>();

    sink->locked_backend()->add_stream(ostream);
    sink->locked_backend()->auto_flush(autoFlush);
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
    return sink;
}

Logging::Logging() {
    boost::log::add_common_attributes();
}

void
Logging::addTextSink(std::shared_ptr<std::ostream> &ostream,
                     LogSeverity minSeverity, bool autoFlush) {
    boost::shared_ptr<std::ostream> boostPtr = boost::shared_ptr<std::ostream>(
        ostream.get(),
        [ostream](std::ostream *) mutable { ostream.reset(); });
    addTextSinkBoostPtr(boostPtr, minSeverity, autoFlush);
}

void Logging::addClog(LogSeverity severity) {
    boost::shared_ptr<std::ostream> stream(&std::clog, boost::null_deleter());
    this->clogSink = addTextSinkBoostPtr(stream, severity, false);
}

void Logging::setClogLevel(LogSeverity level) {
    if(this->clogSink == nullptr) {
        this->addClog(level);
    } else {
        this->clogSink->set_filter(severity >= level);
    }
}

Logger::Handle Logging::getLogger() {
    return std::make_unique<LoggerImpl>();
}

Logger::Handle
Logging::getLogger(const std::vector<arrus::Logger::Attribute> &attributes) {
    return std::make_unique<LoggerImpl>(attributes);
}


}
