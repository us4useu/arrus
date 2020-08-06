#include "logging.h"

namespace arrus {

std::shared_ptr<LoggerFactory> loggerFactory;
Logger::SharedHandle defaultLogger;

void setLoggerFactory(const std::shared_ptr<LoggerFactory>& factory) {
    loggerFactory = factory;
    defaultLogger = factory->getLogger();
}

std::shared_ptr<LoggerFactory> getLoggerFactory() {
    return loggerFactory;
}

Logger::SharedHandle getDefaultLogger() {
    return defaultLogger;
}

}