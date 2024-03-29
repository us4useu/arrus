#include "arrus/core/api/common/LogSeverity.h"
#include "arrus/core/api/common/macros.h"

namespace arrus {

ARRUS_CPP_EXPORT
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

