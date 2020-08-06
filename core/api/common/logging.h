#ifndef ARRUS_CORE_API_COMMON_LOGGING_H
#define ARRUS_CORE_API_COMMON_LOGGING_H


#include <memory>

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
    void setLoggerFactory(const std::shared_ptr<LoggerFactory>& factory);
}

#endif //ARRUS_CORE_COMMON_LOGGING_H
