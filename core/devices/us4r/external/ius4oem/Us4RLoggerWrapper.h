#ifndef ARRUS_CORE_US4R_EXTERNAL_IUS4OEM_US4RLOGGERWRAPPER_H
#define ARRUS_CORE_US4R_EXTERNAL_IUS4OEM_US4RLOGGERWRAPPER_H

#include <unordered_map>

// us4r
#include <logging/Logger.h>

#include "arrus/core/api/common/Logger.h"

namespace arrus {

class Us4RLoggerWrapper : public us4r::Logger {
public:

    Us4RLoggerWrapper(const arrus::Logger::SharedHandle &logger)
            : logger(logger) {}

    void
    log(const us4r::LogSeverity severity, const std::string &msg) override {
        logger->log(sevMap.at(severity), msg);
    }

    void
    setAttribute(const std::string &key, const std::string &value) override {
        logger->setAttribute(key, value);
    }

private:
    arrus::Logger::SharedHandle logger;

    static const inline std::unordered_map<us4r::LogSeverity, arrus::LogSeverity> sevMap{
            {us4r::LogSeverity::TRACE,   arrus::LogSeverity::TRACE},
            {us4r::LogSeverity::DEBUG,   arrus::LogSeverity::DEBUG},
            {us4r::LogSeverity::INFO,    arrus::LogSeverity::INFO},
            {us4r::LogSeverity::WARNING, arrus::LogSeverity::WARNING},
            {us4r::LogSeverity::ERROR,   arrus::LogSeverity::ERROR},
            {us4r::LogSeverity::FATAL,   arrus::LogSeverity::FATAL},
    };
};

}

#endif //ARRUS_CORE_US4R_EXTERNAL_IUS4OEM_US4RLOGGERWRAPPER_H
