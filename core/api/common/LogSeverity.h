#ifndef ARRUS_CORE_API_COMMON_LOGSEVERITY_H
#define ARRUS_CORE_API_COMMON_LOGSEVERITY_H

#include <ostream>

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

#endif //ARRUS_CORE_API_COMMON_LOGSEVERITY_H
