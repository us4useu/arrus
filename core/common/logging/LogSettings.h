#ifndef ARRUS_CORE_COMMON_LOGGING_LOGSETTINGS_H
#define ARRUS_CORE_COMMON_LOGGING_LOGSETTINGS_H

#include <memory>
#include <string>

#include "LogSeverity.h"

namespace arrus {

/**
 * Log settings used in the arrus package.
 */
class LogSettings {
public:
    /**
     * Returns a singleton instance of the log settings container.
     *
     * @return an instance of Log Settings.
     */
    static LogSettings &getInstance() {
        static LogSettings instance;
        return instance;
    }

    /**
     * Adds a given given filename
     *
     * @param filename a path to the output log file
     * @param severity severity level of the records that will be stored in the
     * given output file
     */
    void addLogFile(const std::string &filename, LogSeverity severity);

    /**
     * Sets a minimum severity level for messages printed to the standard output.
     *
     * @param severity severity level to apply
     */
    void setConsoleLogLevel(LogSeverity severity);

    LogSettings(LogSettings const &) = delete;
    void operator=(LogSettings const &) = delete;
    LogSettings(LogSettings const &&) = delete;
    void operator=(LogSettings const &&) = delete;
private:
    LogSettings();
};
}

#endif //ARRUS_CORE_COMMON_LOGGING_LOGSETTINGS_H
