#include "arrus/core/common/logging/LogSeverity.h"

namespace arrus {

std::ostream &operator<<(std::ostream &stream, arrus::LogSeverity level) {
    static const char *enumStrs[] =
            {
                    "TRACE",
                    "DEBUG",
                    "INFO",
                    "WARNING",
                    "ERROR",
                    "FATAL"
            };

    if((unsigned) (level) < sizeof(enumStrs) / sizeof(*enumStrs)) {
        stream << enumStrs[(unsigned int) level];
    } else {
        stream << static_cast<unsigned>(level);
    }
    return stream;
}

}

