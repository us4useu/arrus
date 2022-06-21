#ifndef ARRUS_CORE_COMMON_LOGGING_H
#define ARRUS_CORE_COMMON_LOGGING_H

#include "arrus/core/api/common/LoggerFactory.h"
#include "arrus/core/api/common/logging.h"

namespace arrus {

extern std::shared_ptr<LoggerFactory> loggerFactory;

std::shared_ptr<LoggerFactory> getLoggerFactory();

Logger::SharedHandle getDefaultLogger();

// Deprecated, prefer using ARRUS_INIT_COMPONENT_LOGGER
#define INIT_ARRUS_DEVICE_LOGGER(logger, devId) \
    logger->setAttribute("DeviceId", devId)

#define ARRUS_INIT_COMPONENT_LOGGER(logger, componentId) \
    logger->setAttribute("ComponentId", componentId)

#define ARRUS_LOG(logger, severity, msg) \
    (logger)->log(severity, msg)

#define ARRUS_LOG_DEFAULT(severity, msg) \
    getDefaultLogger()->log(severity, msg)

#define DEFAULT_TEST_LOG_LEVEL arrus::LogSeverity::TRACE

#define ARRUS_INIT_TEST_LOG_LEVEL(ComponentType, level) \
do{                       \
    auto loggingMechanism = std::make_shared<ComponentType>(); \
    loggingMechanism->addClog(level); \
    arrus::setLoggerFactory(loggingMechanism); \
} while(0)
}

#define ARRUS_INIT_TEST_LOG(ComponentType) \
    ARRUS_INIT_TEST_LOG_LEVEL(ComponentType, DEFAULT_TEST_LOG_LEVEL)
#endif //ARRUS_CORE_COMMON_LOGGING_H
