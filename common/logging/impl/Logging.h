#ifndef ARRUS_COMMON_LOGGING_IMPL_LOGGING_H
#define ARRUS_COMMON_LOGGING_IMPL_LOGGING_H

#include <memory>
#include <string>

#include <boost/core/null_deleter.hpp>
#include <boost/log/core.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>

#include "arrus/core/api/common/LogSeverity.h"
#include "arrus/core/api/common/LoggerFactory.h"
#include "arrus/common/logging/impl/LoggerImpl.h"

namespace arrus {

/**
 * Log settings used in the arrus package.
 */
class Logging: public LoggerFactory {
public:
    Logging();
    /**
     * Adds a given given filename
     *
     * @param filename a path to the output log file
     * @param severity severity level of the records that will be stored in the
     * given output file
     */
    void addTextSink(std::shared_ptr<std::ostream> &ostream, LogSeverity severity,
                     bool autoFlush = false);

    /**
     * Sets a minimum severity level for messages printed to the standard output.
     *
     * @param severity severity level to apply
     */
    void addClog(LogSeverity severity);

    /**
     * Sets logging level for clog.
     *
     * Adds clog if necessary.
     *
     * @param level level to set
     */
    void setClogLevel(LogSeverity level);

    Logger::Handle getLogger() override;

    Logger::Handle
    getLogger(const std::vector<arrus::Logger::Attribute> &attributes) override;


    Logging(Logging const &) = delete;
    void operator=(Logging const &) = delete;
    Logging(Logging const &&) = delete;
    void operator=(Logging const &&) = delete;
private:
    boost::shared_ptr<boost::log::sinks::synchronous_sink<
        boost::log::sinks::text_ostream_backend>> clogSink;
};
}

#endif //ARRUS_COMMON_LOGGING_IMPL_LOGGING_H
