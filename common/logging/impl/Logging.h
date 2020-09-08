#ifndef ARRUS_COMMON_LOGGING_IMPL_LOGGING_H
#define ARRUS_COMMON_LOGGING_IMPL_LOGGING_H

#include <memory>
#include <string>

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


    Logger::Handle getLogger() override;

    Logger::Handle
    getLogger(const std::vector<arrus::Logger::Attribute> &attributes) override;


    Logging(Logging const &) = delete;
    void operator=(Logging const &) = delete;
    Logging(Logging const &&) = delete;
    void operator=(Logging const &&) = delete;
private:
};
}

#endif //ARRUS_COMMON_LOGGING_IMPL_LOGGING_H
