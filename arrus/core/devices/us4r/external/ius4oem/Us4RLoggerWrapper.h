#ifndef ARRUS_CORE_US4R_EXTERNAL_IUS4OEM_US4RLOGGERWRAPPER_H
#define ARRUS_CORE_US4R_EXTERNAL_IUS4OEM_US4RLOGGERWRAPPER_H

#include <format>
#include <unordered_map>
#include <utility>
#include <stdexcept>

// us4r
#include <logging/Logger.h>

#include <std4us/concepts.h>
#include <std4us/string.h>

#include "arrus/core/api/common/Logger.h"

namespace arrus::devices {

class Us4RLoggerWrapper : public ::us4us::us4r::Logger {
public:

    explicit Us4RLoggerWrapper(arrus::Logger::SharedHandle logger)
            : logger(std::move(logger)) {}

    void
    log(const ::us4us::us4r::LogSeverity severity, const std::string &msg) override {
        switch(severity) {
        case ::us4us::us4r::LogSeverity::TRACE:
            logger->log(arrus::LogSeverity::TRACE, msg);
            break;
        case ::us4us::us4r::LogSeverity::DEBUG:
            logger->log(arrus::LogSeverity::DEBUG, msg);
            break;
        case ::us4us::us4r::LogSeverity::INFO:
            logger->log(arrus::LogSeverity::INFO, msg);
            break;
        case ::us4us::us4r::LogSeverity::WARNING:
            logger->log(arrus::LogSeverity::WARNING, msg);
            break;
        case ::us4us::us4r::LogSeverity::ERROR:
            logger->log(arrus::LogSeverity::ERROR, msg);
            break;
        case ::us4us::us4r::LogSeverity::FATAL:
            logger->log(arrus::LogSeverity::FATAL, msg);
            break;
        default:
            throw std::runtime_error("Unknown logging level");
        }
    }

    template<typename... Args> requires (std::formattable<Args, char> && ...)
    void log(const ::us4us::us4r::LogSeverity severity, const std::string &fmt, Args... args) {
        log(severity, std::vformat(fmt, std::make_format_args(args...)));
    }

    template<typename... Args> requires ((!std::formattable<Args, char> && std4us::supports_to_string<Args>) && ...)
    void log(const ::us4us::us4r::LogSeverity severity, const std::string &fmt, Args... args) {
        log(severity, std::vformat(fmt, std::make_format_args(std4us::to_string(args)...)));
    }

    template<typename... Args>
    void trace(const std::string &fmt, Args... args) {
        log(::us4us::us4r::LogSeverity::TRACE, fmt, args...);
    }

    template<typename... Args>
    void debug(const std::string &fmt, Args... args) {
        log(::us4us::us4r::LogSeverity::DEBUG, fmt, args...);
    }

    template<typename... Args>
    void info(const std::string &fmt, Args... args) {
        log(::us4us::us4r::LogSeverity::INFO, fmt, args...);
    }

    template<typename... Args>
    void warn(const std::string &fmt, Args... args) {
        log(::us4us::us4r::LogSeverity::WARNING, fmt, args...);
    }

    template<typename... Args>
    void error(const std::string &fmt, Args... args) {
        log(::us4us::us4r::LogSeverity::ERROR, fmt, args...);
    }

    template<typename... Args> requires (std::formattable<Args, char> && ...)
    void fatal(const std::string &fmt, Args... args) {
        log(::us4us::us4r::LogSeverity::FATAL, fmt, args...);
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
