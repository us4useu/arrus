#ifndef ARRUS_CORE_COMMON_LOGGING_H
#define ARRUS_CORE_COMMON_LOGGING_H

#include "arrus/core/api/common/LoggerFactory.h"
#include "arrus/core/api/common/logging.h"

namespace arrus {

extern std::shared_ptr<LoggerFactory> loggerFactory;

std::shared_ptr<LoggerFactory> getLoggerFactory();

Logger::SharedHandle getDefaultLogger();

#define CREATE_ARRUS_DEVICE_LOGGER(devId) \
    getLoggerFactory()->getLogger({{"DeviceId", devId}})

#define ARRUS_LOG(logger, severity, msg) \
    (logger)->log(severity, msg)

#define ARRUS_LOG_DEFAULT(severity, msg) \
    getDefaultLogger()->log(severity, msg)

#define DEFAULT_TEST_LOG_LEVEL arrus::LogSeverity::DEBUG

#define INIT_ARRUS_TEST_LOG(ComponentType) \
do{                       \
    auto loggingMechanism = std::make_shared<ComponentType>(); \
    loggingMechanism->addClog(DEFAULT_TEST_LOG_LEVEL); \
    arrus::setLoggerFactory(loggingMechanism); \
} while(0)
}

#endif //ARRUS_CORE_COMMON_LOGGING_H
