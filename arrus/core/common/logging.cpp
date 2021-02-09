#include <iostream>
#include "logging.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/common/logging/impl/Logging.h"

namespace arrus {

std::shared_ptr<LoggerFactory> loggerFactory;
Logger::SharedHandle defaultLogger;

void setLoggerFactory(const std::shared_ptr<LoggerFactory>& factory) {
    loggerFactory = factory;
    defaultLogger = factory->getLogger();
}

std::shared_ptr<LoggerFactory> getDefaultBoostLoggerFactory() {
    auto loggingMechanism = std::make_shared<::arrus::Logging>();
    std::shared_ptr<std::ostream> ostream{
        std::shared_ptr<std::ostream>(&std::cout, [](std::ostream *) {})};
    loggingMechanism->addTextSink(ostream, ::arrus::LogSeverity::INFO);
    return loggingMechanism;
}

std::shared_ptr<LoggerFactory> getLoggerFactory() {
    if(loggerFactory == nullptr) {
        std::cout << "Using default logging mechanism." << std::endl;
        setLoggerFactory(getDefaultBoostLoggerFactory());
    }
    return loggerFactory;
}

Logger::SharedHandle getDefaultLogger() {
    if(defaultLogger == nullptr) {
        std::cout << "Using default logging mechanism." << std::endl;
        setLoggerFactory(getDefaultBoostLoggerFactory());
    }
    return defaultLogger;
}



}