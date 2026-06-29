#ifndef ARRUS_CORE_API_COMMON_LOGGING_H
#define ARRUS_CORE_API_COMMON_LOGGING_H

#include <boost/core/null_deleter.hpp>
#include <boost/log/core.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <utility>

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
        class LoggingImpl {
            public:
                LoggingImpl();

                void addTextSink(std::shared_ptr<std::ostream> ostream, LogSeverity minSeverity, bool autoFlush);

                void addLogFile(const std::string &filepath, LogSeverity minSeverity);

                void addClog(::arrus::LogSeverity level);

                void setClogLevel(::arrus::LogSeverity level);

                Logger::Handle getLogger();

                Logger::Handle getLogger(const std::vector<arrus::Logger::Attribute> &attributes);
            private:
                boost::shared_ptr<boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>> 
                    clogSink;
                std::set<std::filesystem::path> registeredFiles;
            };

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
         * Adds a log file to the default logging mechanism.
         * If the file has already been added, the call is ignored.
         *
         * NOTE:
         *
         * - This method does not allow to change the logging level of already added log file.
         * This method willy simply ignore the level parameter for any sub-sequent calls of this method for the given
         * file.
         * - This method intentionally DOES NOT EXPAND the '~' (home) directory,
         * as it is currently not explicitly supported by ARRUS logging.
         * This may, however, be supported in the future.
         *
         * @param filepath path to the log file
         * @param level minimum severity level to set for the log file
         */
        ARRUS_CPP_EXPORT
        void addLogFile(const std::string &filepath, LogSeverity level);

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
