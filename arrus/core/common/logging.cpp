#include <iostream>

#include <memory>
#include <string>

#include <boost/core/null_deleter.hpp>
#include <boost/log/core.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <utility>

#include "logging.h"
#include "arrus/core/api/common/exceptions.h"
#include "LoggerImpl.h"

BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", arrus::LogSeverity)
BOOST_LOG_ATTRIBUTE_KEYWORD(deviceIdLogAttr, "DeviceId", std::string)
BOOST_LOG_ATTRIBUTE_KEYWORD(componentIdLogAttr, "ComponentId", std::string)


namespace arrus {

std::shared_ptr<LoggerFactory> getDefaultLoggerFactoryWithClog();

// Global variables.
std::shared_ptr<LoggerFactory> loggerFactory;
Logger::SharedHandle defaultLogger;

// API
void setLoggerFactory(const std::shared_ptr<LoggerFactory>& factory) {
    loggerFactory = factory;
    defaultLogger = factory->getLogger();
}

Logging* useDefaultLoggerFactory() {
    auto pimpl = std::make_unique<Logging::LoggingImpl>();
    loggerFactory = std::make_shared<::arrus::Logging>(std::move(pimpl));
    return (Logging*)loggerFactory.get();
}

std::shared_ptr<LoggerFactory> getLoggerFactory() {
    if(loggerFactory == nullptr) {
        std::cout << "Using default logging mechanism." << std::endl;
        setLoggerFactory(getDefaultLoggerFactoryWithClog());
    }
    return loggerFactory;
}

Logger::SharedHandle getDefaultLogger() {
    if(defaultLogger == nullptr) {
        std::cout << "Using default logging mechanism." << std::endl;
        setLoggerFactory(getDefaultLoggerFactoryWithClog());
    }
    return defaultLogger;
}

// Internal.

// LoggingImpl.
typedef boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend> textSink;

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
        << expr::if_(expr::has_attr(componentIdLogAttr))
               [
                                          expr::stream << "[" << componentIdLogAttr << "]"
    ]
        << " " << severity << ": "
        << expr::smessage;
    sink->set_formatter(formatter);
    boost::log::core::get()->add_sink(sink);
    return sink;
}

class Logging::LoggingImpl {
public:
    LoggingImpl() {
        boost::log::add_common_attributes();
    }

    void addTextSink(std::shared_ptr<std::ostream> ostream, LogSeverity minSeverity, bool autoFlush) {
        boost::shared_ptr<std::ostream> boostPtr = boost::shared_ptr<std::ostream>(
            ostream.get(),
            [ostream](std::ostream *) mutable { ostream.reset(); });
        addTextSinkBoostPtr(boostPtr, minSeverity, autoFlush);
    }

    void addClog(LogSeverity level) {
        boost::shared_ptr<std::ostream> stream(&std::clog, boost::null_deleter());
        this->clogSink = addTextSinkBoostPtr(stream, level, false);
    }

    void setClogLevel(LogSeverity level) {
        if(this->clogSink == nullptr) {
            this->addClog(level);
        } else {
            this->clogSink->set_filter(severity >= level);
        }
    }

    Logger::Handle getLogger() {
        return std::make_unique<LoggerImpl>();
    }

    Logger::Handle getLogger(const std::vector<arrus::Logger::Attribute> &attributes) {
        return std::make_unique<LoggerImpl>(attributes);
    }
private:
    boost::shared_ptr<boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>> clogSink;
};

// Logging.
Logging::Logging(std::unique_ptr<LoggingImpl> pImpl) : pImpl(std::move(pImpl)) {}

Logger::Handle Logging::getLogger() {
    return this->pImpl->getLogger();
}

Logger::Handle Logging::getLogger(const std::vector<arrus::Logger::Attribute> &attributes) {
    return this->pImpl->getLogger(attributes);
}

void Logging::addClog(::arrus::LogSeverity level) {
    this->pImpl->addClog(level);
}

void Logging::addOutputStream(std::shared_ptr<std::ostream> stream, LogSeverity level) {
    this->pImpl->addTextSink(std::move(stream), level, true);
}

void Logging::removeAllStreams() {
    boost::log::core::get()->remove_all_sinks();
}

// Utility functions.
std::shared_ptr<LoggerFactory> getDefaultLoggerFactoryWithClog() {
    auto loggingMechanism = std::make_shared<::arrus::Logging>(std::make_unique<Logging::LoggingImpl>());
    // Do no try to remove std::cout on ptr deletion.
    std::shared_ptr<std::ostream> ostream{std::shared_ptr<std::ostream>(&std::cout, [](std::ostream *) {})};
    loggingMechanism->addOutputStream(ostream, ::arrus::LogSeverity::INFO);
    return loggingMechanism;
}


}