#ifndef ARRUS_CORE_API_COMMON_LOGGING_H
#define ARRUS_CORE_API_COMMON_LOGGING_H

#include <memory>

#include "arrus/core/api/common/macros.h"
#include "arrus/core/api/common/LoggerFactory.h"

namespace arrus {
    /**
     * Sets a logger factory in arrus package.
     *
     * The provided logger factory will be used to generate
     * default and component specific loggers. The logger factory
     * should be available through the life-time of the application.
     *
     * @param factory logger factory to set
     */
    ARRUS_CPP_EXPORT
    void setLoggerFactory(const std::shared_ptr<LoggerFactory>& factory);

    /**
     * Default ARRUS logging mechanism.
     */
    class Logging: public LoggerFactory {
    public:
        class LoggingImpl;

        explicit Logging(std::unique_ptr<LoggingImpl> pImpl);

        ARRUS_CPP_EXPORT
        Logger::Handle getLogger() override;
        ARRUS_CPP_EXPORT
        Logger::Handle getLogger(const std::vector<arrus::Logger::Attribute> &attributes) override;

        ~Logging() override = default;

        /**
         * Adds std::cout logging output stream to the default logging mechanism
         * (console log output).
         *
         * @param level minimum level severity level to set for clog output
         */
        ARRUS_CPP_EXPORT
        void addClog(::arrus::LogSeverity level);

        ARRUS_CPP_EXPORT
        void setClogLevel(::arrus::LogSeverity level);

        /**
         * Adds a custom stream implementation to the default logging mechanism.
         *
         * @param stream output stream to use in logging
         * @param level minimum level severity level to set for the output stream logging
         */
        ARRUS_CPP_EXPORT
        void addOutputStream(std::shared_ptr<std::ostream> stream, LogSeverity level);

        /**
         * Remove all registered output streams from the logging mechanism.
         */
        ARRUS_CPP_EXPORT
        void removeAllStreams();

    private:
        std::unique_ptr<LoggingImpl> pImpl;
    };

    /**
     * Sets default logger factory to ::arrus::Logging.
     *
     * @return raw pointer to the default logging factory.
     */
    ARRUS_CPP_EXPORT
    Logging* useDefaultLoggerFactory();
}

#endif //ARRUS_CORE_COMMON_LOGGING_H
