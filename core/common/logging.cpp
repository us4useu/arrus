#include "logging.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus {

std::shared_ptr<LoggerFactory> loggerFactory;
Logger::SharedHandle defaultLogger;

void setLoggerFactory(const std::shared_ptr<LoggerFactory>& factory) {
    loggerFactory = factory;
    defaultLogger = factory->getLogger();
}

std::shared_ptr<LoggerFactory> getLoggerFactory() {
    if(loggerFactory == nullptr) {
        throw IllegalStateException("Logging mechanism is not initialized, "
                                    "register logger factory first.");
    }
    return loggerFactory;
}

Logger::SharedHandle getDefaultLogger() {
    if(loggerFactory == nullptr) {
        throw IllegalStateException("Logging mechanism is not initialized, "
                                    "register logger factory first.");
    }
    return defaultLogger;
}

}