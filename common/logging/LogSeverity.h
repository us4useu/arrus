#ifndef ARRUS_COMMON_LOGGING_LOGSEVERITY_H
#define ARRUS_COMMON_LOGGING_LOGSEVERITY_H

#include <iostream>

namespace arrus {

enum class LogSeverity {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};
std::ostream &operator<<(std::ostream &stream, arrus::LogSeverity level);

}

#endif //ARRUS_COMMON_LOGGING_LOGSEVERITY_H
