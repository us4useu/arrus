#ifndef ARRUS_CORE_US4R_EXTERNAL_IUS4OEM_US4RLOGGERWRAPPER_H
#define ARRUS_CORE_US4R_EXTERNAL_IUS4OEM_US4RLOGGERWRAPPER_H

#include <unordered_map>
#include <utility>
#include <stdexcept>

// us4r
#include <logging/Logger.h>

#include "arrus/core/api/common/Logger.h"

namespace arrus::devices {

class Us4RLoggerWrapper : public ::us4r::Logger {
public:

    explicit Us4RLoggerWrapper(arrus::Logger::SharedHandle logger)
            : logger(std::move(logger)) {}

    void
    log(const ::us4r::LogSeverity severity, const std::string &msg) override {
        switch(severity) {
            case ::us4r::LogSeverity::TRACE:
                logger->log(arrus::LogSeverity::TRACE, msg);
                break;
            case ::us4r::LogSeverity::DEBUG:
                logger->log(arrus::LogSeverity::DEBUG, msg);
                break;
            case ::us4r::LogSeverity::INFO:
                logger->log(arrus::LogSeverity::INFO, msg);
                break;
            case ::us4r::LogSeverity::WARNING:
                logger->log(arrus::LogSeverity::WARNING, msg);
                break;
            case ::us4r::LogSeverity::ERROR:
                logger->log(arrus::LogSeverity::ERROR, msg);
                break;
            case ::us4r::LogSeverity::FATAL:
                logger->log(arrus::LogSeverity::FATAL, msg);
                break;
            default:
                throw std::runtime_error("Unknown logging level");
        }
    }

    void
    setAttribute(const std::string &key, const std::string &value) override {
        logger->setAttribute(key, value);
    }
private:
    arrus::Logger::SharedHandle logger;
};

}

#endif //ARRUS_CORE_US4R_EXTERNAL_IUS4OEM_US4RLOGGERWRAPPER_H
